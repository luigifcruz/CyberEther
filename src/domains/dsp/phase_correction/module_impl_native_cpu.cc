#include <cmath>

#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/tools/automatic_iterator.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct PhaseCorrectionImplNativeCpu : public PhaseCorrectionImpl,
                                      public NativeCpuRuntimeContext,
                                      public Scheduler::Context {
    Result validate() final;
    Result create() final;
    Result computeSubmit() override;

 private:
    U64 batchCount = 1;
    U64 batchInnerSize = 1;
    U64 channelCount = 1;
    U64 channelInnerSize = 1;
    std::vector<F64> phases;
    std::vector<F64> wrappedPhaseIncrements;
    std::vector<CF32> corrections;
};

Result PhaseCorrectionImplNativeCpu::validate() {
    JST_CHECK(PhaseCorrectionImpl::validate());

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_PHASE_CORRECTION_NATIVE_CPU] Input must be CF32.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result PhaseCorrectionImplNativeCpu::create() {
    JST_CHECK(PhaseCorrectionImpl::create());

    batchCount = batchAxis ? input.shape(*batchAxis) : 1;
    batchInnerSize = 1;
    if (batchAxis) {
        for (Index axis = *batchAxis + 1; axis < input.rank(); ++axis) {
            batchInnerSize *= input.shape(axis);
        }
    }
    channelCount = channelAxis ? input.shape(*channelAxis) : 1;
    channelInnerSize = 1;
    if (channelAxis) {
        for (Index axis = *channelAxis + 1; axis < input.rank(); ++axis) {
            channelInnerSize *= input.shape(axis);
        }
    }

    phases.assign(channelCount, 0.0);
    wrappedPhaseIncrements.resize(channelCount);
    for (U64 channel = 0; channel < channelCount; ++channel) {
        const F64 increment = channelPhaseIncrements.empty()
            ? phaseIncrement
            : channelPhaseIncrements[channel];
        wrappedPhaseIncrements[channel] =
            std::remainder(increment, 2.0 * JST_PI);
    }
    corrections.resize(channelCount * batchCount);
    return Result::SUCCESS;
}

Result PhaseCorrectionImplNativeCpu::computeSubmit() {
    for (U64 channel = 0; channel < channelCount; ++channel) {
        for (U64 batch = 0; batch < batchCount; ++batch) {
            const F64 batchPhase = phases[channel] +
                wrappedPhaseIncrements[channel] * static_cast<F64>(batch);
            corrections[channel * batchCount + batch] = {
                static_cast<F32>(std::cos(batchPhase)),
                static_cast<F32>(std::sin(batchPhase)),
            };
        }
    }

    U64 index = 0;
    JST_CHECK(AutomaticIterator<const CF32, CF32>(
        [&index, this](const auto& in, auto& out) {
            const U64 batch = batchCount == 1
                ? 0
                : (index / batchInnerSize) % batchCount;
            const U64 channel = channelCount == 1
                ? 0
                : (index / channelInnerSize) % channelCount;
            out = in * corrections[channel * batchCount + batch];
            ++index;
        },
        input,
        output));

    for (U64 channel = 0; channel < channelCount; ++channel) {
        phases[channel] = std::remainder(
            phases[channel] +
                wrappedPhaseIncrements[channel] * static_cast<F64>(batchCount),
            2.0 * JST_PI);
    }
    return Result::SUCCESS;
}

JST_REGISTER_MODULE(PhaseCorrectionImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
