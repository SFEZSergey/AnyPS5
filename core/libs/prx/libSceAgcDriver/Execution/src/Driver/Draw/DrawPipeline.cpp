#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include <algorithm>
#include <cstdlib>
#include <string>

namespace AgcDriver::DriverDetail {
namespace {

// Queued jobs hold their draw's prepared data; past this many the worker waits for the recorder.
constexpr std::size_t MaxQueuedJobs = 32;

bool Profiled() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}

void Count(std::vector<std::pair<const char*, std::uint64_t>>& rows, const char* reason) {
    for (auto& row : rows) {
        if (row.first == reason) {
            ++row.second;
            return;
        }
    }
    rows.emplace_back(reason, 1);
}

}

DrawPipeline::DrawPipeline(std::uint32_t queue, std::function<void()> setup) : queue(queue), setup(std::move(setup)) {
    thread = std::thread([this] { run(); });
}

DrawPipeline::~DrawPipeline() {
    {
        std::lock_guard lock(mutex);
        stopping = true;
    }
    wake.notify_all();
    if (thread.joinable()) thread.join();
}

void DrawPipeline::run() {
    if (setup) setup();
    std::unique_lock lock(mutex);
    for (;;) {
        if (!stopping && jobs.empty()) {
            recorderWaiting = true;
            wake.wait(lock, [&] { return stopping || !jobs.empty(); });
            recorderWaiting = false;
        }
        if (jobs.empty()) return;
        auto job = std::move(jobs.front());
        jobs.pop_front();
        running = std::move(job.writes);
        lock.unlock();
        const bool timed = Profiled();
        const auto start = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        std::exception_ptr error;
        try {
            job.work();
        } catch (...) {
            error = std::current_exception();
        }
        job.work = nullptr;
        if (timed) recordedNs.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()), std::memory_order_relaxed);
        lock.lock();
        running.clear();
        if (error != nullptr && failure == nullptr) failure = error;
        pending.fetch_sub(1, std::memory_order_acq_rel);
        if ((waitingForRoom && jobs.size() <= MaxQueuedJobs / 2) || (waitingForDrain && jobs.empty()) || failure != nullptr) done.notify_all();
    }
}

void DrawPipeline::Enqueue(Job job) {
    std::unique_lock lock(mutex);
    if (failure != nullptr) std::rethrow_exception(failure);
    if (jobs.size() >= MaxQueuedJobs) {
        const auto start = Profiled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        waitingForRoom = true;
        done.wait(lock, [&] { return jobs.size() <= MaxQueuedJobs / 2 || failure != nullptr; });
        waitingForRoom = false;
        if (failure != nullptr) std::rethrow_exception(failure);
        if (Profiled()) {
            ++fullWaits;
            fullWaitMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
    }
    jobs.push_back(std::move(job));
    pending.fetch_add(1, std::memory_order_acq_rel);
    ++enqueued;
    const bool signal = recorderWaiting;
    lock.unlock();
    if (signal) wake.notify_one();
    if (Profiled()) report();
}

void DrawPipeline::Drain(const char* reason) {
    std::unique_lock lock(mutex);
    ++drains;
    if (pending.load(std::memory_order_acquire) != 0) {
        const auto start = Profiled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        waitingForDrain = true;
        done.wait(lock, [&] { return pending.load(std::memory_order_acquire) == 0; });
        waitingForDrain = false;
        if (Profiled()) {
            ++drainsWaited;
            drainWaitMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            Count(drainReasons, reason);
        }
    }
    if (failure != nullptr) {
        auto error = failure;
        failure = nullptr;
        std::rethrow_exception(error);
    }
}

bool DrawPipeline::Overlaps(std::span<const ShaderRecompiler::MemoryRegion> regions) {
    if (Idle()) return false;
    std::lock_guard lock(mutex);
    const auto overlaps = [&](const std::vector<Range>& writes) {
        for (const auto& [begin, end] : writes) {
            for (const auto& region : regions) {
                if (region.guestAddress < end && begin < region.guestAddress + region.bytes.size()) return true;
            }
        }
        return false;
    };
    if (overlaps(running)) return true;
    return std::any_of(jobs.begin(), jobs.end(), [&](const Job& job) { return overlaps(job.writes); });
}

void DrawPipeline::NoteSyncDraw(const char* reason) {
    if (!Profiled()) return;
    std::lock_guard lock(mutex);
    Count(syncReasons, reason);
}

void DrawPipeline::report() {
    std::lock_guard lock(mutex);
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport < std::chrono::seconds(10)) return;
    const auto seconds = std::chrono::duration<double>(now - lastReport).count();
    lastReport = now;
    std::string drainText, syncText;
    for (const auto& [reason, count] : drainReasons) drainText += std::string(" ") + reason + " " + std::to_string(count);
    for (const auto& [reason, count] : syncReasons) syncText += std::string(" ") + reason + " " + std::to_string(count);
    const auto recorded = static_cast<double>(recordedNs.exchange(0, std::memory_order_relaxed)) / 1e6;
    AgcDriver::ProfilePrint_nid_no_patch("[pipeline] queue 0x%x (%.0f s): %llu draws to the recorder (busy %.0f ms, %.0f%%), %llu drains (%llu waited %.1f ms; by reason:%s), full queue %llu x %.1f ms; draws kept on the worker:%s\n",
        queue, seconds, static_cast<unsigned long long>(enqueued), recorded, recorded / (seconds * 10.0), static_cast<unsigned long long>(drains), static_cast<unsigned long long>(drainsWaited), drainWaitMs, drainText.empty() ? " none" : drainText.c_str(),
        static_cast<unsigned long long>(fullWaits), fullWaitMs, syncText.empty() ? " none" : syncText.c_str());
    enqueued = drains = drainsWaited = fullWaits = 0;
    drainWaitMs = fullWaitMs = 0;
    drainReasons.clear();
    syncReasons.clear();
}

}
