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
    Result validate() final;

    Result presentInitialize() override;
    Result presentSubmit() override;
    Result computeSubmit() override;
};

Result WaterfallImplNativeCpu::validate() {
    JST_CHECK(WaterfallImpl::validate());

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() != DataType::F32) {
        JST_ERROR("[MODULE_WATERFALL_NATIVE_CPU] Unsupported input data type: {}.",
                  inputTensor.dtype());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result WaterfallImplNativeCpu::presentInitialize() {
    return createPresent();
}

Result WaterfallImplNativeCpu::presentSubmit() {
    return present();
}

Result WaterfallImplNativeCpu::computeSubmit() {
    const auto plan = PlanWaterfallWrite(ringState.writeIndex,
                                         numberOfBatches,
                                         height);

    // Copy input data to frequency bins buffer (circular buffer pattern).

    F32* freqData = static_cast<F32*>(frequencyBins.data());
    const F32* inputData = static_cast<const F32*>(input.data());

    for (U64 row = 0; row < plan.rowCount; ++row) {
        const U64 sourceBatch = plan.sourceRow + row;
        const U64 destinationBatch = (plan.destinationRow + row) % height;
        for (U64 column = 0; column < numberOfElements; ++column) {
            freqData[destinationBatch * numberOfElements + column] =
                inputData[sourceBatch * inputBatchStride +
                          column * inputSampleStride];
        }
    }

    // Update write index.

    ringState.advance(numberOfBatches, height);

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(WaterfallImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
