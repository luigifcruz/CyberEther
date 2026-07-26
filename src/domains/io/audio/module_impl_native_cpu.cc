#include <limits>

#include <jetstream/memory/macros.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct AudioImplNativeCpu : public AudioImpl,
                            public NativeCpuRuntimeContext,
                            public Scheduler::Context {
 public:
    Result validate() final;

    Result computeSubmit() override;
};

Result AudioImplNativeCpu::validate() {
    JST_CHECK(AudioImpl::validate());

    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& inputBuffer = inputs().at("buffer").tensor;
    if (!inputBuffer.validShape() || inputBuffer.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputBuffer.dtype() != DataType::F32) {
        JST_ERROR("[MODULE_AUDIO_NATIVE_CPU] Input buffer must be F32.");
        return Result::ERROR;
    }

    U64 alignedOutputSize = 0;
    if (!detail::CheckedPageAlignedSize(validatedOutputSizeBytes,
                                        alignedOutputSize) ||
        alignedOutputSize > std::numeric_limits<std::size_t>::max() ||
        validatedCircularBufferSizeBytes >
            std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_AUDIO_NATIVE_CPU] Output or circular buffer "
                  "allocation size is too large.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result AudioImplNativeCpu::computeSubmit() {
    return resample();
}

JST_REGISTER_MODULE(AudioImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
