#include <cmath>
#include <complex>
#include <algorithm>
#include <limits>

#include <jetstream/backend/devices/cpu/helpers.hh>
#include <jetstream/memory/macros.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

static constexpr F64 kPi = JST_PI;

struct PskDemodImplNativeCpu : public PskDemodImpl,
                               public NativeCpuRuntimeContext,
                               public Scheduler::Context {
 public:
    Result validate() final;
    Result computeSubmit() override;
};

Result PskDemodImplNativeCpu::validate() {
    JST_CHECK(PskDemodImpl::validate());

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_PSK_DEMOD_NATIVE_CPU] Input must be complex (CF32).");
        return Result::ERROR;
    }

    U64 alignedOutputSize = 0;
    if (!detail::CheckedPageAlignedSize(validatedOutputSizeBytes, alignedOutputSize) ||
        alignedOutputSize > std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_PSK_DEMOD_NATIVE_CPU] Output allocation size is too large.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result PskDemodImplNativeCpu::computeSubmit() {
    const CF32* inputData = input.data<CF32>();
    CF32* outputData = output.data<CF32>();

    // Early return for empty output buffers.
    if (outputSize == 0) {
        return Result::SUCCESS;
    }

    const U64 inputSampleStride = input.stride(sampleAxis);
    const U64 outputSampleStride = output.stride(sampleAxis);
    const U64 inputBatchStride = batchAxis ? input.stride(*batchAxis) : 0;
    const U64 outputBatchStride = batchAxis ? output.stride(*batchAxis) : 0;

    for (U64 lane = 0; lane < laneCount; ++lane) {
        U64 inputLaneOffset = 0;
        U64 outputLaneOffset = 0;
        U64 remainder = lane;
        for (auto axis = laneAxes.rbegin(); axis != laneAxes.rend(); ++axis) {
            const U64 coordinate = remainder % input.shape(*axis);
            remainder /= input.shape(*axis);
            inputLaneOffset += coordinate * input.stride(*axis);
            outputLaneOffset += coordinate * output.stride(*axis);
        }

        auto& state = laneStates[lane];
        for (U64 batch = 0; batch < batchSize; ++batch) {
            const U64 inputOffset = inputLaneOffset + batch * inputBatchStride;
            const U64 outputOffset = outputLaneOffset + batch * outputBatchStride;

            for (U64 sample = 0; sample < inputSampleSize; ++sample) {
                state.sampleHistory.push_back(
                    inputData[inputOffset + sample * inputSampleStride]);
            }

            U64 outputIndex = 0;
            F64 mu = state.timingMu;
            F64 omega = state.timingOmega;
            U64 index = state.timingIndex;
            F64 phase = state.phaseAccumulator;
            F64 freqAcc = state.frequencyError;
            bool hasPrevSymbol = state.hasLastSymbol;
            CF32 prevSymbol = state.lastSymbol;
            CF32 prevDecision = state.lastDecision;
            U64 iterations = 0;

            while (outputIndex < outputSampleSize && iterations < maxIterations) {
                ++iterations;
                const U64 historySize =
                    static_cast<U64>(state.sampleHistory.size());

                while (mu >= 1.0 && index + 1 < historySize) {
                    mu -= 1.0;
                    ++index;
                }
                while (mu < 0.0 && index > 0) {
                    mu += 1.0;
                    --index;
                }
                if (mu < 0.0) {
                    mu = 0.0;
                }
                if (index + 1 >= historySize) {
                    break;
                }

                const CF32& earlySample = state.sampleHistory[index];
                const CF32& lateSample = state.sampleHistory[index + 1];
                const CF32 interpolatedSample =
                    interpolate(earlySample, lateSample, mu);
                const CF32 corrected = correctFrequency(interpolatedSample, phase);
                const CF32 decisionPoint = decision(corrected);

                if (hasPrevSymbol) {
                    F64 timingErr = muellerMullerError(
                        prevSymbol, prevDecision, corrected, decisionPoint);
                    timingErr = std::clamp(
                        timingErr, MIN_TIMING_ERROR, MAX_TIMING_ERROR);
                    omega += timingBeta * timingErr;
                    omega = std::clamp(omega, timingOmegaMin, timingOmegaMax);
                    mu += timingAlpha * timingErr;
                }

                const F64 freqErrSample = costasLoopError(corrected);
                freqAcc += freqBeta * freqErrSample;
                freqAcc = std::clamp(freqAcc, -kPi, kPi);
                phase += freqAcc + freqAlpha * freqErrSample;
                phase = std::remainder(phase, 2.0 * kPi);

                outputData[outputOffset + outputIndex * outputSampleStride] =
                    corrected;
                ++outputIndex;
                prevSymbol = corrected;
                prevDecision = decisionPoint;
                hasPrevSymbol = true;
                mu += omega;
            }

            const std::size_t historySize = state.sampleHistory.size();
            std::size_t pruneCount = 0;
            if (historySize > 1) {
                pruneCount = std::min<std::size_t>(
                    static_cast<std::size_t>(index), historySize - 1);
                for (std::size_t i = 0; i < pruneCount; ++i) {
                    state.sampleHistory.pop_front();
                }
            }
            if (pruneCount > 0) {
                index -= static_cast<U64>(pruneCount);
            }

            state.timingMu = mu;
            state.timingOmega = omega;
            state.timingIndex = index;
            state.phaseAccumulator = phase;
            state.frequencyError = freqAcc;
            state.hasLastSymbol = hasPrevSymbol;
            state.lastSymbol = prevSymbol;
            state.lastDecision = prevDecision;

            while (outputIndex < outputSampleSize) {
                outputData[outputOffset + outputIndex * outputSampleStride] =
                    CF32{0.0f, 0.0f};
                ++outputIndex;
            }
        }
    }

    // Keep the legacy observable field synchronized with the first lane.
    frequencyError = laneStates.empty() ? 0.0 : laneStates.front().frequencyError;

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(PskDemodImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
