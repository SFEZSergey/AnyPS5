// Нативный (aarch64) shim ядра AnyPS5 для группы, загружаемой box64 (см. Aps5Native.hpp): определяет
// функции libc/libkernel, нужные драйверу и выводу, и переадресует их в x86-ядро через мост box64.
// C++-объекты x86-стороны (shared_ptr) сюда не попадают: x86 держит их у себя и отдаёт дескрипторы,
// а нативные shared_ptr освобождают их через x86 при последней ссылке.
#include "prx/aps5native/Aps5Native.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include "prx/libkernel/AppMetadata/include/AppMetadata.hpp"
#include "prx/libkernel/Equeue/Equeue.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <stop_token>
#include <vector>

namespace {

Aps5Native::X86Caller caller = nullptr;
std::array<std::uint64_t, Aps5Native::X86FunctionCount> functions{};
std::stop_source shutdownSource;

std::uint64_t x86(Aps5Native::X86Function function, std::uint64_t a0 = 0, std::uint64_t a1 = 0, std::uint64_t a2 = 0, std::uint64_t a3 = 0, std::uint64_t a4 = 0, std::uint64_t a5 = 0) {
    if (caller == nullptr || functions[function] == 0) {
        std::fprintf(stderr, "[aps5native] x86 function %u called before Aps5CoreShimInit\n", static_cast<unsigned>(function));
        std::abort();
    }
    return caller(static_cast<std::uintptr_t>(functions[function]), a0, a1, a2, a3, a4, a5);
}

template <class T>
std::uint64_t arg(T value) {
    if constexpr (std::is_pointer_v<T>) return reinterpret_cast<std::uintptr_t>(value);
    else return static_cast<std::uint64_t>(value);
}

bool asBool(std::uint64_t value) {
    return (value & 0xffu) != 0;
}

// Отпущенные нативной стороной x86-закрепления копятся и уходят в x86 одним вызовом: переход в x86
// (box64 RunFunction: sigsetjmp + sigprocmask) на каждый диапазон каждого снимка съедал до четверти
// потока драйвера. Очередь сбрасывается при следующем снимке, при переполнении и перед ожиданием
// x86-ядра на закреплённом диапазоне (pin waiter ниже), так что гость не ждёт отложенных.
std::mutex unpinMutex;
std::vector<void*> unpinQueue;
constexpr std::size_t UnpinBatchLimit = 4096;

bool flushUnpins() {
    std::vector<void*> batch;
    {
        std::lock_guard lock(unpinMutex);
        batch.swap(unpinQueue);
    }
    if (batch.empty()) return false;
    x86(Aps5Native::RangeUnpinBatch, arg(batch.data()), batch.size());
    return true;
}

void queueUnpin(void* pin) {
    bool full = false;
    {
        std::lock_guard lock(unpinMutex);
        unpinQueue.push_back(pin);
        full = unpinQueue.size() >= UnpinBatchLimit;
    }
    if (full) flushUnpins();
}

bool (*nativePinWaiter)(std::uintptr_t, std::size_t) = nullptr;

// Снимок реестра, переиспользуемый, пока поколение реестра не изменилось: драйвер берёт снимок на
// каждую отрисовку (HostImportFor и др.), а полный снимок — переход в x86 и по выделению на диапазон.
std::mutex leaseCacheMutex;
GuestAllocations::Lease leaseCache;
std::uint64_t leaseCacheGeneration = ~std::uint64_t{0};

bool dropLeaseCache() {
    GuestAllocations::Lease dropped;
    {
        std::lock_guard lock(leaseCacheMutex);
        dropped.swap(leaseCache);
        leaseCacheGeneration = ~std::uint64_t{0};
    }
    return !dropped.empty();
}

bool pinWaiterFlushing(std::uintptr_t address, std::size_t bytes) {
    // Кэш снимка держит закрепления всех диапазонов: x86 ждёт освобождения — отпускаем его первым.
    const bool dropped = dropLeaseCache();
    // Сброшенные отложенные закрепления — уже продвижение: x86 перепроверит диапазон сразу.
    if (flushUnpins() || dropped) return true;
    return nativePinWaiter != nullptr && nativePinWaiter(address, bytes);
}

// Снимок x86 (Lease в куче x86) как нативный Lease: у каждого элемента своё x86-закрепление,
// которое отпускается при последней нативной ссылке на этот диапазон; сам снимок отпускается сразу.
GuestAllocations::Lease adoptLease(void* lease) {
    GuestAllocations::Lease result;
    if (lease == nullptr) return result;
    flushUnpins();
    const auto count = x86(Aps5Native::LeaseCount, arg(lease));
    std::vector<const GuestAllocations::Range*> ranges(count);
    std::vector<void*> pins(count);
    if (count != 0) x86(Aps5Native::LeasePinAll, arg(lease), arg(ranges.data()), arg(pins.data()));
    x86(Aps5Native::LeaseRelease, arg(lease));
    result.reserve(count);
    for (std::uint64_t i = 0; i < count; ++i) {
        void* pin = pins[i];
        result.emplace_back(ranges[i], [pin](const GuestAllocations::Range*) { queueUnpin(pin); });
    }
    return result;
}

void (*registeredShutdown[8])() = {};
int registeredShutdownCount = 0;

// Обработчик остановки libc (x86 вызывает его через мост): останавливает нативный токен и зовёт
// обработчики группы.
void onShutdown() {
    shutdownSource.request_stop();
    for (int i = 0; i < registeredShutdownCount; ++i) {
        try {
            registeredShutdown[i]();
        } catch (...) {
        }
    }
}

void dispatchGpuMap(void* callback, void* lease, std::uint64_t generation) {
    const auto pinned = adoptLease(lease);
    const GuestAllocations::Mapped mapped(pinned.begin(), pinned.end());
    try {
        reinterpret_cast<void (*)(const GuestAllocations::Mapped&, std::uint64_t)>(callback)(mapped, generation);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[aps5native] GPU map observer failed: %s\n", error.what());
    } catch (...) {
    }
}

} // namespace

extern "C" void Aps5CoreShimInit(Aps5Native::X86Caller x86Caller, const std::uint64_t* table, std::uint64_t count) {
    caller = x86Caller;
    if (count != Aps5Native::X86FunctionCount) {
        std::fprintf(stderr, "[aps5native] x86 function table has %llu entries, expected %u\n", static_cast<unsigned long long>(count), static_cast<unsigned>(Aps5Native::X86FunctionCount));
        std::abort();
    }
    for (std::uint64_t i = 0; i < count; ++i) functions[i] = table[i];
    if (std::getenv("APS5_NATIVE_DEBUG") != nullptr) {
        for (std::uint64_t i = 0; i < count; ++i) std::fprintf(stderr, "[aps5native] x86 function %llu = 0x%llx\n", static_cast<unsigned long long>(i), static_cast<unsigned long long>(functions[i]));
    }
    x86(Aps5Native::RegisterShutdown, arg(&onShutdown));
}

// --- libkernel: очереди событий ---------------------------------------------------------------

extern "C" KernelEqueueRef EqueuePin_nid_postfix(KernelEqueue eq) {
    void* handle = reinterpret_cast<void*>(x86(Aps5Native::EqueuePinAcquire, arg(eq)));
    if (handle == nullptr) return {};
    // Объект очереди нативный код не трогает: указатель лишь отмечает «очередь жива».
    auto* object = *static_cast<KernelEqueuePrivate**>(handle);
    return KernelEqueueRef(object, [handle](KernelEqueuePrivate*) { x86(Aps5Native::EqueuePinRelease, arg(handle)); });
}

extern "C" int EqueueAddEvent_nid_postfix(KernelEqueue eq, const KernelEqueueEvent& event) {
    // Колбэки фильтра вызывает x86-ядро: подменяем их x86-трамплинами. Владельца (нативный
    // shared_ptr) x86 освобождать не может: держим его здесь до конца процесса.
    KernelEqueueEvent copy = event;
    const auto trampoline = [](Aps5Native::EqueueCallback kind, void* native) {
        return reinterpret_cast<void*>(x86(Aps5Native::EqueueTrampoline, kind, arg(native)));
    };
    copy.filter.triggerFunc = reinterpret_cast<EqueueTriggerFunc>(trampoline(Aps5Native::EqueueTrigger, reinterpret_cast<void*>(event.filter.triggerFunc)));
    copy.filter.resetFunc = reinterpret_cast<EqueueResetFunc>(trampoline(Aps5Native::EqueueReset, reinterpret_cast<void*>(event.filter.resetFunc)));
    copy.filter.deleteEventFunc = reinterpret_cast<EqueueDeleteFunc>(trampoline(Aps5Native::EqueueDelete, reinterpret_cast<void*>(event.filter.deleteEventFunc)));
    copy.filter.pollFunc = reinterpret_cast<EqueuePollFunc>(trampoline(Aps5Native::EqueuePoll, reinterpret_cast<void*>(event.filter.pollFunc)));
    if (copy.filter.owner) {
        static std::mutex ownersMutex;
        static std::vector<std::shared_ptr<void>> owners;
        std::lock_guard lock(ownersMutex);
        owners.push_back(std::move(copy.filter.owner));
    }
    return static_cast<int>(x86(Aps5Native::EqueueAddEvent, arg(eq), arg(&copy)));
}

extern "C" int EqueueTriggerEvent_nid_postfix(KernelEqueue eq, uintptr_t ident, int16_t filter, void* triggerData) {
    return static_cast<int>(x86(Aps5Native::EqueueTriggerEvent, arg(eq), ident, static_cast<std::uint64_t>(static_cast<std::int64_t>(filter)), arg(triggerData)));
}

extern "C" void EqueuePushPending_nid_postfix(KernelEqueueEvent* event, const KernelEvent* pending) {
    x86(Aps5Native::EqueuePushPending, arg(event), arg(pending));
}

extern "C" int EqueueDeleteEvent_nid_postfix(KernelEqueue eq, uintptr_t ident, int16_t filter) {
    return static_cast<int>(x86(Aps5Native::EqueueDeleteEvent, arg(eq), ident, static_cast<std::uint64_t>(static_cast<std::int64_t>(filter))));
}

// --- libc: арена и регистр выделений гостя -----------------------------------------------------

namespace GuestArena {
extern "C" void GuestArenaRange_nid_postfix(std::uintptr_t* base, std::size_t* bytes) {
    x86(Aps5Native::GuestArenaRange, arg(base), arg(bytes));
}
extern "C" bool GuestArenaSharedBacking_nid_postfix(std::uintptr_t address, std::size_t bytes, int* file, std::uint64_t* offset) {
    return asBool(x86(Aps5Native::GuestArenaSharedBacking, address, bytes, arg(file), arg(offset)));
}
extern "C" bool GuestArenaWriteSharedBacking_nid_no_patch(std::uintptr_t address, const void* source, std::size_t bytes) {
    return asBool(x86(Aps5Native::GuestArenaWriteSharedBacking, address, arg(source), bytes));
}
} // namespace GuestArena

namespace GuestAllocations {
extern "C" Lease GuestAllocationsAcquire_nid_postfix() {
    // Опыт: кэш держит закрепления всех диапазонов, и x86-ядро ждёт их при каждом освобождении памяти
    // гостя (титульник 60 -> 20 FPS). Оставлен выключенным: APS5_LEASE_CACHE=1 — включить.
    static const bool cached = std::getenv("APS5_LEASE_CACHE") != nullptr;
    if (!cached) return adoptLease(reinterpret_cast<void*>(x86(Aps5Native::LeaseAcquire)));
    // Поколение читается до снимка: изменение между ними даст новый ключ и новый снимок при следующем вызове.
    const auto generation = x86(Aps5Native::GuestAllocationsGeneration);
    {
        std::lock_guard lock(leaseCacheMutex);
        if (generation == leaseCacheGeneration) return leaseCache;
    }
    auto lease = adoptLease(reinterpret_cast<void*>(x86(Aps5Native::LeaseAcquire)));
    GuestAllocations::Lease previous;
    {
        std::lock_guard lock(leaseCacheMutex);
        previous.swap(leaseCache);
        leaseCache = lease;
        leaseCacheGeneration = generation;
    }
    return lease;
}
extern "C" std::uint64_t GuestAllocationsReadableEnd_nid_postfix(std::uintptr_t begin, std::uintptr_t end) {
    return x86(Aps5Native::ReadableEnd, begin, end);
}
extern "C" std::uint64_t GuestAllocationsGeneration_nid_postfix() {
    return x86(Aps5Native::GuestAllocationsGeneration);
}
extern "C" void GuestAllocationsSetInvalidator_nid_postfix(void (*callback)(std::uintptr_t address, std::size_t bytes)) {
    x86(Aps5Native::SetInvalidator, arg(callback));
}
extern "C" void GuestAllocationsSetPinWaiter_nid_postfix(bool (*callback)(std::uintptr_t address, std::size_t bytes)) {
    nativePinWaiter = callback;
    x86(Aps5Native::SetPinWaiter, arg(callback != nullptr ? &pinWaiterFlushing : nullptr));
}
extern "C" void GuestAllocationsSetGpuMapObserver_nid_postfix(void (*callback)(const Mapped& ranges, std::uint64_t generation)) {
    x86(Aps5Native::SetGpuMapObserver, arg(&dispatchGpuMap), arg(callback));
}
} // namespace GuestAllocations

// --- libc: отслеживание записей — нативная реализация в WriteWatchNative.cpp ---------------------

// --- libc: завершение процесса -----------------------------------------------------------------

extern "C" void LibcRegisterShutdown_nid_postfix(void (*callback)()) {
    if (registeredShutdownCount < static_cast<int>(std::size(registeredShutdown))) registeredShutdown[registeredShutdownCount++] = callback;
}

extern "C" std::stop_token LibcShutdownToken_nid_postfix() {
    if (!shutdownSource.stop_requested() && caller != nullptr && asBool(x86(Aps5Native::ShutdownRequested))) shutdownSource.request_stop();
    return shutdownSource.get_token();
}

extern "C" void LibcRequestShutdown_nid_postfix() {
    x86(Aps5Native::LibcRequestShutdown);
}

extern "C" void LibcRequestExit_nid_postfix(int code) {
    x86(Aps5Native::LibcRequestExit, static_cast<std::uint64_t>(static_cast<std::int64_t>(code)));
}

extern "C" [[noreturn]] void LibcAwaitExit_nid_postfix() {
    x86(Aps5Native::LibcAwaitExit);
    std::abort();
}

extern "C" void NotImplemented_nid_no_patch(const char* funcName) {
    x86(Aps5Native::NotImplemented, arg(funcName));
}

// --- libkernel: время и метаданные приложения --------------------------------------------------

extern "C" std::uint64_t sceKernelGetProcessTime() {
    return x86(Aps5Native::ProcessTime);
}

extern "C" std::uint64_t sceKernelGetProcessTimeCounter() {
    return x86(Aps5Native::ProcessTimeCounter);
}

extern "C" std::uint64_t sceKernelGetProcessTimeCounterFrequency() {
    static const std::uint64_t frequency = x86(Aps5Native::ProcessTimeCounterFrequency);
    return frequency;
}

extern "C" bool HasAppIcon_nid_postfix() {
    return asBool(x86(Aps5Native::HasAppIcon));
}

extern "C" AppTitle GetAppTitle_nid_postfix() {
    AppTitle title{};
    x86(Aps5Native::AppTitle, arg(&title));
    return title;
}

extern "C" AppIconData GetAppIconData_nid_postfix() {
    AppIconData data{};
    x86(Aps5Native::AppIconData, arg(&data));
    return data;
}

extern "C" void* mmap_nid_postfix(void* address, std::size_t length, int protection, int flags, int descriptor, std::int64_t offset) noexcept {
    return reinterpret_cast<void*>(x86(Aps5Native::Mmap, arg(address), length, static_cast<std::uint64_t>(static_cast<std::int64_t>(protection)), static_cast<std::uint64_t>(static_cast<std::int64_t>(flags)), static_cast<std::uint64_t>(static_cast<std::int64_t>(descriptor)), static_cast<std::uint64_t>(offset)));
}
