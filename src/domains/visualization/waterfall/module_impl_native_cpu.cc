#include <algorithm>

#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct WaterfallImplNativeCpu : public WaterfallImpl,
                                public NativeCpuRuntimeContext,
                                public Scheduler::Context {
 public:
    Result create() final;

    Result presentInitialize() override;
    Result presentSubmit() override;
    Result computeSubmit() override;

 private:
    U64 writeIndex = 0;
};

Result WaterfallImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(WaterfallImpl::create());

    // Validate input dtype.

    if (input.dtype() != DataType::F32) {
        JST_ERROR("[MODULE_WATERFALL_NATIVE_CPU] Unsupported input data type: {}.", input.dtype());
        return Result::ERROR;
    }

    // Initialize state.

    writeIndex = 0;

    return Result::SUCCESS;
}

Result WaterfallImplNativeCpu::presentInitialize() {
    return createPresent();
}

Result WaterfallImplNativeCpu::presentSubmit() {
    return present();
}

Result WaterfallImplNativeCpu::computeSubmit() {
    const auto totalSize = input.size();
    const auto fftSize = numberOfElements;
    const auto offset = writeIndex * fftSize;
    const auto size = std::min(totalSize, (height - writeIndex) * fftSize);

    // Copy input data to frequency bins buffer (circular buffer pattern).

    F32* freqData = static_cast<F32*>(frequencyBins.data());
    const F32* inputData = static_cast<const F32*>(input.data());

    std::copy(inputData, inputData + size, freqData + offset);

    if (size < totalSize) {
        std::copy(inputData + size, inputData + totalSize, freqData);
    }

    // Update write index.

    writeIndex = (writeIndex + numberOfBatches) % height;
    inc = writeIndex;

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(WaterfallImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
