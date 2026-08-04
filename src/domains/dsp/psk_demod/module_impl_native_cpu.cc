#include <array>
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

    const auto laneOffset = [&](const Tensor& tensor, const U64 lane) {
        U64 offset = 0;
        U64 remainder = lane;
        for (auto axis = laneAxes.rbegin(); axis != laneAxes.rend(); ++axis) {
            const U64 coordinate = remainder % tensor.shape(*axis);
            remainder /= tensor.shape(*axis);
            offset += coordinate * tensor.stride(*axis);
        }
        return offset;
    };

    const auto hasCompleteOutput = [&]() {
        return std::all_of(laneStates.begin(), laneStates.end(), [&](const auto& state) {
            return state.pendingSymbols.size() >= outputSymbolsPerLane;
        });
    };

    const auto emitOutput = [&]() -> Result {
        for (U64 lane = 0; lane < laneCount; ++lane) {
            const U64 outputLaneOffset = laneOffset(output, lane);
            auto& pendingSymbols = laneStates[lane].pendingSymbols;
            for (U64 batch = 0; batch < batchSize; ++batch) {
                const U64 outputOffset =
                    outputLaneOffset + batch * outputBatchStride;
                JST_CHECK(pendingSymbols.popStrided(outputData + outputOffset,
                                                    outputSampleSize,
                                                    outputSampleStride));
            }
        }
        return Result::SUCCESS;
    };

    const bool emittedQueuedOutput = hasCompleteOutput();
    if (emittedQueuedOutput) {
        JST_CHECK(emitOutput());
    }

    for (U64 lane = 0; lane < laneCount; ++lane) {
        const U64 inputLaneOffset = laneOffset(input, lane);

        auto& state = laneStates[lane];
        for (U64 batch = 0; batch < batchSize; ++batch) {
            const U64 inputOffset = inputLaneOffset + batch * inputBatchStride;

            JST_CHECK(state.sampleHistory.pushStrided(
                inputData + inputOffset, inputSampleSize, inputSampleStride));

            F64 mu = state.timingMu;
            F64 omega = state.timingOmega;
            U64 index = state.timingIndex;
            F64 phase = state.phaseAccumulator;
            F64 freqAcc = state.frequencyError;
            bool hasPrevSymbol = state.hasLastSymbol;
            CF32 prevSymbol = state.lastSymbol;
            CF32 prevDecision = state.lastDecision;
            U64 iterations = 0;
            const U64 historySize = state.sampleHistory.size();
            bool historyExhausted = false;

            while (iterations < maxIterations) {
                ++iterations;

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
                    historyExhausted = true;
                    break;
                }

                std::array<CF32, 2> interpolationSamples;
                JST_CHECK(state.sampleHistory.peek(
                    index, interpolationSamples.data(), interpolationSamples.size()));
                const CF32 interpolatedSample =
                    interpolate(interpolationSamples[0], interpolationSamples[1], mu);
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

                if (state.pendingSymbols.push(&corrected, 1) != Result::SUCCESS) {
                    JST_ERROR("[MODULE_PSK_DEMOD_NATIVE_CPU] Pending symbol capacity was exceeded.");
                    return Result::ERROR;
                }
                prevSymbol = corrected;
                prevDecision = decisionPoint;
                hasPrevSymbol = true;
                mu += omega;
            }

            if (!historyExhausted) {
                JST_ERROR("[MODULE_PSK_DEMOD_NATIVE_CPU] Timing recovery exceeded its iteration limit.");
                return Result::ERROR;
            }

            U64 pruneCount = 0;
            if (historySize > 1) {
                pruneCount = std::min(index, historySize - 1);
                JST_CHECK(state.sampleHistory.discard(pruneCount));
            }
            if (pruneCount > 0) {
                index -= pruneCount;
            }

            state.timingMu = mu;
            state.timingOmega = omega;
            state.timingIndex = index;
            state.phaseAccumulator = phase;
            state.frequencyError = freqAcc;
            state.hasLastSymbol = hasPrevSymbol;
            state.lastSymbol = prevSymbol;
            state.lastDecision = prevDecision;
        }
    }

    // Keep the legacy observable field synchronized with the first lane.
    frequencyError = laneStates.empty() ? 0.0 : laneStates.front().frequencyError;

    if (emittedQueuedOutput) {
        return Result::SUCCESS;
    }
    if (!hasCompleteOutput()) {
        return Result::SKIP;
    }
    return emitOutput();
}

JST_REGISTER_MODULE(PskDemodImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
