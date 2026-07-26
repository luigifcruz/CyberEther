#include <cmath>

#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct SpectrogramImplNativeCpu : public SpectrogramImpl,
                                  public NativeCpuRuntimeContext,
                                  public Scheduler::Context {
 public:
    Result validate() final;
    Result create() final;

    Result presentInitialize() override;
    Result presentSubmit() override;
    Result computeSubmit() override;
};

Result SpectrogramImplNativeCpu::validate() {
    JST_CHECK(SpectrogramImpl::validate());

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() != DataType::F32) {
        JST_ERROR("[MODULE_SPECTROGRAM_NATIVE_CPU] Unsupported input data type: {}.",
                  inputTensor.dtype());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result SpectrogramImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(SpectrogramImpl::create());
    return Result::SUCCESS;
}

Result SpectrogramImplNativeCpu::presentInitialize() {
    return createPresent();
}

Result SpectrogramImplNativeCpu::presentSubmit() {
    return present();
}

Result SpectrogramImplNativeCpu::computeSubmit() {
    const U64 totalBins = frequencyBins.size();
    F32* freqData = static_cast<F32*>(frequencyBins.data());
    const F32* inputData = static_cast<const F32*>(input.data());

    // Apply decay to all frequency bins.

    for (U64 i = 0; i < totalBins; ++i) {
        freqData[i] *= decayFactor;
    }

    // Accumulate new data into frequency bins.

    for (U64 b = 0; b < numberOfBatches; ++b) {
        for (U64 x = 0; x < numberOfElements; ++x) {
            const U64 index = static_cast<U64>(inputData[b * numberOfElements + x] * height);

            if (index > 0 && index < height) {
                F32& val = freqData[x + (index * numberOfElements)];
                val = std::min(val + 0.02f, 1.0f);
            }
        }
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(SpectrogramImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
