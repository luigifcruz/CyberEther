#include <limits>

#include <jetstream/memory/macros.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct SoapyImplNativeCpu : public SoapyImpl,
                            public NativeCpuRuntimeContext,
                            public Scheduler::Context {
 public:
    Result validate() final;

    Result computeSubmit() override;
    Result hasPendingCompute() override;
};

Result SoapyImplNativeCpu::validate() {
    JST_CHECK(SoapyImpl::validate());

    U64 alignedOutputSize = 0;
    if (!detail::CheckedPageAlignedSize(validatedOutputSizeBytes,
                                        alignedOutputSize) ||
        alignedOutputSize > std::numeric_limits<std::size_t>::max() ||
        validatedInternalSizeBytes > std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_SOAPY_NATIVE_CPU] Output or internal buffer "
                  "allocation size is too large.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result SoapyImplNativeCpu::hasPendingCompute() {
    if (circularBuffer.getOccupancy() < buffer.size()) {
        return circularBuffer.waitBufferOccupancy(buffer.size());
    }

    return Result::SUCCESS;
}

Result SoapyImplNativeCpu::computeSubmit() {
    if (errored) {
        return Result::ERROR;
    }

    if (circularBuffer.getOccupancy() < buffer.size()) {
        return Result::YIELD;
    }

    circularBuffer.get(reinterpret_cast<CF32*>(buffer.data()), buffer.size());

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(SoapyImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
