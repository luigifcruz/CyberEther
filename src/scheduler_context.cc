#include "jetstream/scheduler_context.hh"

namespace Jetstream {

Result Scheduler::Context::presentInitialize() {
    return Result::SUCCESS;
}

Result Scheduler::Context::presentSubmit() {
    return Result::SUCCESS;
}

Result Scheduler::Context::hasPendingCompute() {
    return Result::SUCCESS;
}

Result Scheduler::Context::hasPendingPresent() {
    return Result::SUCCESS;
}

}  // namespace Jetstream
