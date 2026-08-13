#include <algorithm>
#include <cmath>

#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct SignalViewImplNativeCpu : public SignalViewImpl,
                                 public NativeCpuRuntimeContext,
                                 public Scheduler::Context {
 public:
    Result validate() final;
    Result create() final;

    Result presentInitialize() override;
    Result presentSubmit() override;
    Result computeSubmit() override;

    Buffer::Config renderStateBufferConfig() const override;
    Result resetAveragingState() override;

 private:
    Tensor sums;
    Tensor averagingBuffer;
};

Result SignalViewImplNativeCpu::validate() {
    JST_CHECK(SignalViewImpl::validate());

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() != DataType::F32) {
        JST_ERROR("[MODULE_SIGNAL_VIEW_NATIVE_CPU] Unsupported input data type: {}.",
                  inputTensor.dtype());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result SignalViewImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(SignalViewImpl::create());

    if (lineplotEnabled) {
        JST_CHECK(sums.create(DeviceType::CPU, DataType::F32, {numberOfElements}));
        JST_CHECK(averagingBuffer.create(DeviceType::CPU,
                                         DataType::F32,
                                         {numberOfElements}));

        detail::InitializeLineplotPoints(
            static_cast<F32*>(signalPoints.data()),
            static_cast<F32*>(maxHoldPoints.data()),
            numberOfElements);
    }

    return Result::SUCCESS;
}

Result SignalViewImplNativeCpu::presentInitialize() {
    return createPresent();
}

Buffer::Config SignalViewImplNativeCpu::renderStateBufferConfig() const {
    return {};
}

Result SignalViewImplNativeCpu::resetAveragingState() {
    if (!lineplotEnabled) {
        return Result::SUCCESS;
    }
    std::fill_n(static_cast<F32*>(averagingBuffer.data()),
                numberOfElements, 0.0f);
    return Result::SUCCESS;
}

Result SignalViewImplNativeCpu::presentSubmit() {
    return present();
}

Result SignalViewImplNativeCpu::computeSubmit() {
    const F32* inputData = static_cast<const F32*>(input.data());

    bool updateMaxHold = false;
    if (lineplotEnabled) {
        F32* sumsData = static_cast<F32*>(sums.data());
        F32* avgData = static_cast<F32*>(averagingBuffer.data());
        F32* signalData = static_cast<F32*>(signalPoints.data());

        std::fill_n(sumsData, numberOfElements, 0.0f);

        for (U64 b = 0; b < numberOfBatches; b++) {
            for (U64 i = 0; i < numberOfElements; i++) {
                sumsData[i] += inputData[detail::LineplotInputIndex(
                    b, i, inputBatchStride, inputSampleStride, decimation)];
            }
        }

        updateMaxHold =
            maxHold && detail::LineplotMaxHoldReady(maxHoldWarmupBlocks, averaging);
        F32* maxData =
            updateMaxHold ? static_cast<F32*>(maxHoldPoints.data()) : nullptr;

        for (U64 i = 0; i < numberOfElements; i++) {
            const auto amplitude = std::fmin(
                std::fmax((sumsData[i] * normalizationFactor) - 1.0f, -1.0f),
                1.0f);

            auto& average = avgData[i];
            average -= average / averaging;
            average += amplitude / averaging;

            signalData[(i * 2) + 1] = average;

            if (maxData) {
                auto& maxVal = maxData[(i * 2) + 1];
                if (average > maxVal) { maxVal = average; }
            }
        }

        if (maxHold && maxHoldWarmupBlocks < averaging) {
            ++maxHoldWarmupBlocks;
        }

        updateSignalPointsFlag = true;
        if (updateMaxHold) { updateHoldPointsFlag = true; }
    }

    if (waterfallEnabled) {
        const auto plan = PlanWaterfallWrite(waterfallHistory.writeIndex,
                                             numberOfBatches,
                                             waterfallHeight);
        F32* waterfallData = static_cast<F32*>(waterfallBins.data());

        for (U64 row = 0; row < plan.rowCount; ++row) {
            const U64 sourceBatch = plan.sourceRow + row;
            const U64 destinationBatch =
                (plan.destinationRow + row) % waterfallHeight;
            for (U64 sample = 0; sample < inputSampleSize; ++sample) {
                waterfallData[destinationBatch * inputSampleSize + sample] =
                    inputData[sourceBatch * inputBatchStride +
                              sample * inputSampleStride];
            }
        }

        waterfallHistory.advance(numberOfBatches, waterfallHeight);
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(SignalViewImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
