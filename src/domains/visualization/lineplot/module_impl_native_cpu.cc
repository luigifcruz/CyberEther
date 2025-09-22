#include <algorithm>

#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct LineplotImplNativeCpu : public LineplotImpl,
                               public Runtime::Context,
                               public Scheduler::Context {
 public:
    Result create() final;

    Result presentInitialize() override;
    Result presentSubmit() override;
    Result computeSubmit() override;

 private:
    Tensor sums;
    Tensor averagingBuffer;
};

Result LineplotImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(LineplotImpl::create());

    // Validate input dtype.

    if (input.dtype() != DataType::F32) {
        JST_ERROR("[MODULE_LINEPLOT_NATIVE_CPU] Unsupported input data type: {}.", input.dtype());
        return Result::ERROR;
    }

    // Allocate averaging buffers.

    JST_CHECK(sums.create(DeviceType::CPU, DataType::F32, {numberOfElements}));
    JST_CHECK(averagingBuffer.create(DeviceType::CPU, DataType::F32, {numberOfElements}));

    // Initialize signal points X coordinates.

    F32* signalData = static_cast<F32*>(signalPoints.data());
    for (U64 i = 0; i < numberOfElements; i++) {
        signalData[(i * 2) + 0] = i * 2.0f / (numberOfElements - 1) - 1.0f;
        signalData[(i * 2) + 1] = 0.0f;
    }

    return Result::SUCCESS;
}

Result LineplotImplNativeCpu::presentInitialize() {
    return createPresent();
}

Result LineplotImplNativeCpu::presentSubmit() {
    return present();
}

Result LineplotImplNativeCpu::computeSubmit() {
    F32* sumsData = static_cast<F32*>(sums.data());
    F32* avgData = static_cast<F32*>(averagingBuffer.data());
    F32* signalData = static_cast<F32*>(signalPoints.data());
    const F32* inputData = static_cast<const F32*>(input.data());

    // Clear sums.

    for (U64 i = 0; i < numberOfElements; i++) {
        sumsData[i] = 0.0f;
    }

    // Sum across batches with decimation.

    for (U64 b = 0; b < numberOfBatches; b++) {
        for (U64 i = 0; i < numberOfElements; i++) {
            sumsData[i] += inputData[(i * decimation) + b * numberOfElements * decimation];
        }
    }

    // Apply normalization and averaging.

    for (U64 i = 0; i < numberOfElements; i++) {
        // Get amplitude.
        const auto amplitude = (sumsData[i] * normalizationFactor) - 1.0f;

        // Calculate moving average.
        auto& average = avgData[i];
        average -= average / averaging;
        average += amplitude / averaging;

        signalData[(i * 2) + 1] = average;
    }

    updateSignalPointsFlag = true;

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(LineplotImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
