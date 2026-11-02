#include "jetstream/scheduler_context.hh"
#include "jetstream/detail/scheduler_impl.hh"

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace Jetstream {

namespace {

struct PendingComputeSignal {
    using Clock = std::chrono::steady_clock;

    std::mutex mutex;
    std::condition_variable cond;
    std::atomic<U64> epoch{0};

    static PendingComputeSignal& Instance() {
        static PendingComputeSignal signal;
        return signal;
    }
};

struct ComputeRound {
    bool busy = false;
    PendingComputeSignal::Clock::time_point deadline = PendingComputeSignal::Clock::time_point::max();
};

thread_local ComputeRound round;

}  // namespace

Result Scheduler::Context::presentInitialize() {
    return Result::SUCCESS;
}

Result Scheduler::Context::presentCapture() {
    return Result::SUCCESS;
}

Result Scheduler::Context::presentSubmit() {
    return Result::SUCCESS;
}

Result Scheduler::Context::hasPendingCompute() {
    return Result::SUCCESS;
}

void Scheduler::Context::NotifyPendingCompute() {
    auto& signal = PendingComputeSignal::Instance();
    {
        std::lock_guard lock(signal.mutex);
        signal.epoch.fetch_add(1, std::memory_order_release);
    }
    signal.cond.notify_all();
}

void Scheduler::Context::NotifyPendingComputeAt(const std::chrono::steady_clock::time_point deadline) {
    Scheduler::Impl::ReportComputeDeadline(deadline);
}

U64 Scheduler::Impl::BeginComputeRound() {
    round = {};
    return PendingComputeSignal::Instance().epoch.load(std::memory_order_acquire);
}

void Scheduler::Impl::ReportComputeProgress() {
    round.busy = true;
}

void Scheduler::Impl::ReportComputeDeadline(const std::chrono::steady_clock::time_point deadline) {
    round.deadline = std::min(round.deadline, deadline);
}

void Scheduler::Impl::WaitComputeRound(const U64 epoch, const std::chrono::milliseconds timeout) {
    using Clock = PendingComputeSignal::Clock;
    auto& signal = PendingComputeSignal::Instance();

    if (round.busy) {
        return;
    }

    const auto until = std::min(Clock::now() + timeout, round.deadline);

    std::unique_lock lock(signal.mutex);
    signal.cond.wait_until(lock, until, [&] {
        return signal.epoch.load(std::memory_order_acquire) != epoch;
    });
}

}  // namespace Jetstream
