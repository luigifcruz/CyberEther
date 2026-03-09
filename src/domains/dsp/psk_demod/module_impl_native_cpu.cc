#include <cmath>
#include <complex>
#include <algorithm>

#include <jetstream/backend/devices/cpu/helpers.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

static constexpr F64 kPi = JST_PI;

struct PskDemodImplNativeCpu : public PskDemodImpl,
                               public Runtime::Context,
                               public Scheduler::Context {
 public:
    Result create() override;
    Result computeSubmit() override;
};

Result PskDemodImplNativeCpu::create() {
    JST_CHECK(PskDemodImpl::create());

    if (input.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_PSK_DEMOD_NATIVE_CPU] Input must be complex (CF32).");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result PskDemodImplNativeCpu::computeSubmit() {
    const CF32* inputData = input.data<CF32>();
    CF32* outputData = output.data<CF32>();
    const U64 inputSize = input.size();

    // Early return for empty output buffers.
    if (outputSize == 0) {
        return Result::SUCCESS;
    }

    // Append newly received samples to the interpolation history.
    for (U64 i = 0; i < inputSize; ++i) {
        sampleHistory.push_back(inputData[i]);
    }

    U64 outputIndex = 0;

    // Local copies of the loop state for better cache behavior.
    F64 mu = timingMu;
    F64 omega = timingOmega;
    U64 index = timingIndex;
    F64 phase = phaseAccumulator;
    F64 freqAcc = frequencyError;
    bool hasPrevSymbol = hasLastSymbol;
    CF32 prevSymbol = lastSymbol;
    CF32 prevDecision = lastDecision;

    // Safety counter prevents infinite loops if configuration is pathological.
    U64 iterations = 0;
    const U64 maxIterations = outputSize * (samplesPerSymbol + 4);

    while (outputIndex < outputSize && iterations < maxIterations) {
        iterations++;

        const U64 historySize = static_cast<U64>(sampleHistory.size());

        // Bring mu back into [0, 1) while staying within the available history.
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

        // Not enough samples yet to interpolate the next symbol.
        if (index + 1 >= historySize) {
            break;
        }

        const CF32& earlySample = sampleHistory[index];
        const CF32& lateSample = sampleHistory[index + 1];
        CF32 interpolatedSample = interpolate(earlySample, lateSample, mu);
        CF32 corrected = correctFrequency(interpolatedSample, phase);
        CF32 decisionPoint = decision(corrected);

        F64 timingErr = 0.0;
        if (hasPrevSymbol) {
            timingErr = muellerMullerError(prevSymbol, prevDecision, corrected, decisionPoint);
            timingErr = std::clamp(timingErr, MIN_TIMING_ERROR, MAX_TIMING_ERROR);
            omega += timingBeta * timingErr;
            omega = std::clamp(omega, timingOmegaMin, timingOmegaMax);
            mu += timingAlpha * timingErr;
        }

        F64 freqErrSample = costasLoopError(corrected);
        freqAcc += freqAlpha * freqErrSample;
        freqAcc = std::clamp(freqAcc, -kPi, kPi);
        phase += freqAcc + freqBeta * freqErrSample;
        phase = std::remainder(phase, 2.0 * kPi);

        outputData[outputIndex++] = corrected;

        prevSymbol = corrected;
        prevDecision = decisionPoint;
        hasPrevSymbol = true;

        mu += omega;
    }

    // Discard the samples that are no longer needed while keeping one look-back sample.
    std::size_t historySize = sampleHistory.size();
    std::size_t pruneCount = 0;
    if (historySize > 1) {
        pruneCount = std::min<std::size_t>(static_cast<std::size_t>(index), historySize - 1);
        for (std::size_t i = 0; i < pruneCount; ++i) {
            sampleHistory.pop_front();
        }
    }
    if (pruneCount > 0) {
        index -= static_cast<U64>(pruneCount);
    }

    // Store state back.
    timingMu = mu;
    timingOmega = omega;
    timingIndex = index;
    phaseAccumulator = phase;
    frequencyError = freqAcc;
    hasLastSymbol = hasPrevSymbol;
    lastSymbol = prevSymbol;
    lastDecision = prevDecision;

    // Zero-fill any remaining output slots to preserve deterministic output sizes.
    // TODO: Create helper to concatenate chucks.

    while (outputIndex < outputSize) {
        outputData[outputIndex++] = CF32{0.0f, 0.0f};
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(PskDemodImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
