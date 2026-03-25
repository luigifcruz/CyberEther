#include "jetstream/runtime_context_native_cpu.hh"

namespace Jetstream {

Result NativeCpuRuntimeContext::computeInitialize() {
    return Result::SUCCESS;
}

Result NativeCpuRuntimeContext::computeSubmit() {
    return Result::SUCCESS;
}

Result NativeCpuRuntimeContext::computeDeinitialize() {
    return Result::SUCCESS;
}

}  // namespace Jetstream
