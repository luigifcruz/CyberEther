#include "jetstream/scheduler_context.hh"
#include "jetstream/detail/scheduler_impl.hh"

#include <algorithm>

#include "jetstream/platform.hh"

namespace Jetstream {

namespace {

using Clock = Platform::DeadlineWaiter::Clock;

Platform::DeadlineWaiter& PendingComputeSignal() {
    static Platform::DeadlineWaiter signal;
    return signal;
}

struct ComputeRound {
    bool busy = false;
    Clock::time_point deadline = Clock::time_point::max();
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
    PendingComputeSignal().notify();
}

void Scheduler::Context::NotifyPendingComputeAt(const std::chrono::steady_clock::time_point deadline) {
    Scheduler::Impl::ReportComputeDeadline(deadline);
}

U64 Scheduler::Impl::BeginComputeRound() {
    round = {};
    return PendingComputeSignal().generation();
}

void Scheduler::Impl::ReportComputeProgress() {
    round.busy = true;
}

void Scheduler::Impl::ReportComputeDeadline(const std::chrono::steady_clock::time_point deadline) {
    round.deadline = std::min(round.deadline, deadline);
}

void Scheduler::Impl::WaitComputeRound(const U64 epoch, const std::chrono::milliseconds timeout) {
    if (round.busy) {
        return;
    }

    const auto until = std::min(Clock::now() + timeout, round.deadline);

    PendingComputeSignal().waitUntil(until, epoch);
}

}  // namespace Jetstream
