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
    F64 phase = 0.0;
    F64 wrappedPhaseIncrement = 0.0;
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

    phase = 0.0;
    wrappedPhaseIncrement = std::remainder(phaseIncrement, 2.0 * JST_PI);
    batchCount = batchAxis ? input.shape(*batchAxis) : 1;
    batchInnerSize = 1;
    if (batchAxis) {
        for (Index axis = *batchAxis + 1; axis < input.rank(); ++axis) {
            batchInnerSize *= input.shape(axis);
        }
    }
    corrections.resize(batchCount);
    return Result::SUCCESS;
}

Result PhaseCorrectionImplNativeCpu::computeSubmit() {
    for (U64 batch = 0; batch < batchCount; ++batch) {
        const F64 batchPhase =
            phase + wrappedPhaseIncrement * static_cast<F64>(batch);
        corrections[batch] = {
            static_cast<F32>(std::cos(batchPhase)),
            static_cast<F32>(std::sin(batchPhase)),
        };
    }

    U64 index = 0;
    const U64 innerSize = batchInnerSize;
    const U64 count = batchCount;
    JST_CHECK(AutomaticIterator<const CF32, CF32>(
        [&index, innerSize, count, this](const auto& in, auto& out) {
            const U64 batch = count == 1 ? 0 : (index / innerSize) % count;
            out = in * corrections[batch];
            ++index;
        },
        input,
        output));

    phase = std::remainder(
        phase + wrappedPhaseIncrement * static_cast<F64>(batchCount),
        2.0 * JST_PI);
    return Result::SUCCESS;
}

JST_REGISTER_MODULE(PhaseCorrectionImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
