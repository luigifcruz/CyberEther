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
};

Result WaterfallImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(WaterfallImpl::create());

    // Validate input dtype.

    if (input.dtype() != DataType::F32) {
        JST_ERROR("[MODULE_WATERFALL_NATIVE_CPU] Unsupported input data type: {}.", input.dtype());
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
    const U64 firstRowCount = std::min(plan.rowCount,
                                       height - plan.destinationRow);

    // Copy input data to frequency bins buffer (circular buffer pattern).

    F32* freqData = static_cast<F32*>(frequencyBins.data());
    const F32* inputData = static_cast<const F32*>(input.data());

    std::copy_n(inputData + plan.sourceRow * numberOfElements,
                firstRowCount * numberOfElements,
                freqData + plan.destinationRow * numberOfElements);

    const U64 secondRowCount = plan.rowCount - firstRowCount;
    if (secondRowCount > 0) {
        std::copy_n(inputData + (plan.sourceRow + firstRowCount) * numberOfElements,
                    secondRowCount * numberOfElements,
                    freqData);
    }

    // Update write index.

    ringState.advance(numberOfBatches, height);

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(WaterfallImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
