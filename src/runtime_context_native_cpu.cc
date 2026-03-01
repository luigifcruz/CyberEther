#include "jetstream/runtime_context_native_cpu.hh"

namespace Jetstream {

Result Runtime::Context::computeInitialize() {
    return Result::SUCCESS;
}

Result Runtime::Context::computeSubmit() {
    return Result::SUCCESS;
}

}  // namespace Jetstream
