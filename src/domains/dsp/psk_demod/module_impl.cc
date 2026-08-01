#include "module_impl.hh"

#include <algorithm>
#include <cmath>
#include <limits>

#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

static constexpr F32 kPi = static_cast<F32>(JST_PI);

Result PskDemodImpl::validate() {
    validatedSamplesPerSymbol = 0;
    validatedConstellationOrder = 0;
    validatedOutputSize = 0;
    validatedOutputSizeBytes = 0;
    validatedMaxIterations = 0;
    validatedInputSampleSize = 0;
    validatedOutputSampleSize = 0;
    validatedSampleHistoryCapacity = 0;
    validatedOutputSymbolsPerLane = 0;
    validatedPendingSymbolsCapacity = 0;
    validatedBatchSize = 1;
    validatedLaneCount = 1;
    validatedSignalAxes = {};
    validatedOutputShape.clear();
    validatedFreqAlpha = 0.0;
    validatedFreqBeta = 0.0;
    validatedTimingAlpha = 0.0;
    validatedTimingBeta = 0.0;
    validatedTimingOmegaNominal = 0.0;
    validatedTimingOmegaMin = 0.0;
    validatedTimingOmegaMax = 0.0;
    validatedOutputSampleRate = 0.0f;

    const auto& config = *candidate();

    if (!std::isfinite(config.sampleRate) || config.sampleRate <= 0.0) {
        JST_ERROR("[MODULE_PSK_DEMOD] Sample rate must be finite and positive.");
        return Result::ERROR;
    }

    if (!std::isfinite(config.symbolRate) || config.symbolRate <= 0.0) {
        JST_ERROR("[MODULE_PSK_DEMOD] Symbol rate must be finite and positive.");
        return Result::ERROR;
    }

    constexpr F64 maxF32 = static_cast<F64>(std::numeric_limits<F32>::max());
    if (config.symbolRate > maxF32) {
        JST_ERROR("[MODULE_PSK_DEMOD] Symbol rate must be representable as nonzero F32 metadata.");
        return Result::ERROR;
    }
    const F32 candidateOutputSampleRate = static_cast<F32>(config.symbolRate);
    if (candidateOutputSampleRate <= 0.0f) {
        JST_ERROR("[MODULE_PSK_DEMOD] Symbol rate must be representable as nonzero F32 metadata.");
        return Result::ERROR;
    }

    if (config.symbolRate >= config.sampleRate) {
        JST_ERROR("[MODULE_PSK_DEMOD] Symbol rate must be less than sample rate.");
        return Result::ERROR;
    }

    if (!std::isfinite(config.frequencyLoopBandwidth) ||
        config.frequencyLoopBandwidth <= 0.0 || config.frequencyLoopBandwidth >= 1.0) {
        JST_ERROR("[MODULE_PSK_DEMOD] Frequency loop bandwidth must be between 0 and 1.");
        return Result::ERROR;
    }

    if (!std::isfinite(config.timingLoopBandwidth) ||
        config.timingLoopBandwidth <= 0.0 || config.timingLoopBandwidth >= 1.0) {
        JST_ERROR("[MODULE_PSK_DEMOD] Timing loop bandwidth must be between 0 and 1.");
        return Result::ERROR;
    }

    if (!std::isfinite(config.dampingFactor) || config.dampingFactor <= 0.0) {
        JST_ERROR("[MODULE_PSK_DEMOD] Damping factor must be finite and positive.");
        return Result::ERROR;
    }

    U64 candidateConstellationOrder = 0;
    if (config.pskType == "bpsk") {
        candidateConstellationOrder = 2;
    } else if (config.pskType == "qpsk") {
        candidateConstellationOrder = 4;
    } else if (config.pskType == "8psk") {
        candidateConstellationOrder = 8;
    } else {
        JST_ERROR("[MODULE_PSK_DEMOD] Unsupported PSK type: {}.", config.pskType);
        return Result::ERROR;
    }

    const F64 candidateTimingOmegaNominal = config.sampleRate / config.symbolRate;
    const F64 samplesPerSymbolLimit = std::ldexp(
        1.0, std::numeric_limits<U64>::digits);
    if (!std::isfinite(candidateTimingOmegaNominal) ||
        candidateTimingOmegaNominal < 2.0 ||
        candidateTimingOmegaNominal >= samplesPerSymbolLimit) {
        JST_ERROR("[MODULE_PSK_DEMOD] Samples per symbol must be at least 2 and representable.");
        return Result::ERROR;
    }

    const U64 candidateSamplesPerSymbol =
        static_cast<U64>(candidateTimingOmegaNominal);
    if (candidateSamplesPerSymbol < 2) {
        JST_ERROR("[MODULE_PSK_DEMOD] Samples per symbol must be at least 2.");
        return Result::ERROR;
    }

    const auto deriveLoopCoefficients = [&](const F64 bandwidth,
                                            F64& alpha,
                                            F64& beta) {
        F64 denominator = 0.0;
        if (config.dampingFactor > std::numeric_limits<F64>::max() / 4.0) {
            denominator = (1.0 + bandwidth * bandwidth) / config.dampingFactor +
                          2.0 * bandwidth;
            alpha = (4.0 * bandwidth) / denominator;
            beta = ((4.0 * bandwidth) * bandwidth / config.dampingFactor) /
                   denominator;
        } else {
            denominator = 1.0 +
                          (2.0 * config.dampingFactor) * bandwidth +
                          bandwidth * bandwidth;
            alpha = ((4.0 * config.dampingFactor) * bandwidth) / denominator;
            beta = ((4.0 * bandwidth) * bandwidth) / denominator;
        }
        return std::isfinite(denominator) && denominator > 0.0 &&
               std::isfinite(alpha) && alpha > 0.0 &&
               std::isfinite(beta) && beta > 0.0;
    };

    F64 candidateFreqAlpha = 0.0;
    F64 candidateFreqBeta = 0.0;
    F64 candidateTimingAlpha = 0.0;
    F64 candidateTimingBeta = 0.0;
    if (!deriveLoopCoefficients(config.frequencyLoopBandwidth,
                                candidateFreqAlpha,
                                candidateFreqBeta) ||
        !deriveLoopCoefficients(config.timingLoopBandwidth,
                                candidateTimingAlpha,
                                candidateTimingBeta)) {
        JST_ERROR("[MODULE_PSK_DEMOD] Loop parameters do not produce usable finite coefficients.");
        return Result::ERROR;
    }

    const F64 candidateTimingOmegaMin =
        std::max(0.5, candidateTimingOmegaNominal * 0.5);
    const F64 candidateTimingOmegaMax =
        std::max(candidateTimingOmegaMin + 1e-6,
                 candidateTimingOmegaNominal * 1.5);
    if (!std::isfinite(candidateTimingOmegaMin) || candidateTimingOmegaMin <= 0.0 ||
        !std::isfinite(candidateTimingOmegaMax) ||
        candidateTimingOmegaNominal < candidateTimingOmegaMin ||
        candidateTimingOmegaNominal > candidateTimingOmegaMax) {
        JST_ERROR("[MODULE_PSK_DEMOD] Timing loop geometry is not usable.");
        return Result::ERROR;
    }

    validatedSamplesPerSymbol = candidateSamplesPerSymbol;
    validatedConstellationOrder = candidateConstellationOrder;
    validatedFreqAlpha = candidateFreqAlpha;
    validatedFreqBeta = candidateFreqBeta;
    validatedTimingAlpha = candidateTimingAlpha;
    validatedTimingBeta = candidateTimingBeta;
    validatedTimingOmegaNominal = candidateTimingOmegaNominal;
    validatedTimingOmegaMin = candidateTimingOmegaMin;
    validatedTimingOmegaMax = candidateTimingOmegaMax;
    validatedOutputSampleRate = candidateOutputSampleRate;

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    SignalAxes axes;
    if (ResolveSignalAxes(inputTensor, axes) != Result::SUCCESS) {
        JST_ERROR("[MODULE_PSK_DEMOD] Input must contain valid signal axis metadata.");
        return Result::ERROR;
    }

    U64 requiredInputBytes = 0;
    if (!detail::CheckedMultiply(inputTensor.size(),
                                 static_cast<U64>(sizeof(CF32)),
                                 requiredInputBytes) ||
        inputTensor.sizeBytes() < requiredInputBytes) {
        JST_ERROR("[MODULE_PSK_DEMOD] Input metadata cannot hold its CF32 samples.");
        return Result::ERROR;
    }

    const Buffer& inputBuffer = inputTensor.buffer();
    const U64 inputCapacityBytes = inputBuffer.sizeBytes();
    if (!inputBuffer.valid() || inputTensor.offsetBytes() > inputCapacityBytes ||
        requiredInputBytes > inputCapacityBytes - inputTensor.offsetBytes()) {
        JST_ERROR("[MODULE_PSK_DEMOD] Input buffer does not cover its CF32 sample range.");
        return Result::ERROR;
    }

    const U64 inputSampleSize = inputTensor.shape(*axes.sample);
    if (inputSampleSize < candidateSamplesPerSymbol) {
        JST_ERROR("[MODULE_PSK_DEMOD] Sample axis is too short to produce any symbols.");
        return Result::ERROR;
    }

    const long double candidateOutputSampleSize =
        static_cast<long double>(inputSampleSize) *
        static_cast<long double>(config.symbolRate) /
        static_cast<long double>(config.sampleRate);
    if (!std::isfinite(candidateOutputSampleSize) ||
        candidateOutputSampleSize <= 0.0L ||
        candidateOutputSampleSize >
            static_cast<long double>(std::numeric_limits<U64>::max())) {
        JST_ERROR("[MODULE_PSK_DEMOD] Output sample geometry is not representable.");
        return Result::ERROR;
    }
    const U64 outputSampleSize =
        static_cast<U64>(std::ceil(candidateOutputSampleSize));

    U64 candidateSampleHistoryCapacity = 0;
    if (!detail::CheckedAdd(inputSampleSize,
                            U64{1},
                            candidateSampleHistoryCapacity) ||
        candidateSampleHistoryCapacity >
            static_cast<U64>(std::numeric_limits<std::size_t>::max())) {
        JST_ERROR("[MODULE_PSK_DEMOD] Sample history capacity exceeds the supported range.");
        return Result::ERROR;
    }

    Shape candidateOutputShape = inputTensor.shape();
    candidateOutputShape[*axes.sample] = outputSampleSize;

    U64 candidateOutputSize = 1;
    for (const U64 dimension : candidateOutputShape) {
        if (!detail::CheckedMultiply(candidateOutputSize,
                                     dimension,
                                     candidateOutputSize)) {
            JST_ERROR("[MODULE_PSK_DEMOD] Output shape exceeds the supported range.");
            return Result::ERROR;
        }
    }

    U64 candidateLaneCount = 1;
    for (Index axis = 0; axis < inputTensor.rank(); ++axis) {
        if (axis == *axes.sample || (axes.batch && axis == *axes.batch)) {
            continue;
        }
        if (!detail::CheckedMultiply(candidateLaneCount,
                                     inputTensor.shape(axis),
                                     candidateLaneCount)) {
            JST_ERROR("[MODULE_PSK_DEMOD] Independent lane count exceeds the supported range.");
            return Result::ERROR;
        }
    }

    U64 candidateOutputSizeBytes = 0;
    if (!detail::CheckedMultiply(candidateOutputSize,
                                 static_cast<U64>(sizeof(CF32)),
                                 candidateOutputSizeBytes) ||
        candidateOutputSizeBytes == 0) {
        JST_ERROR("[MODULE_PSK_DEMOD] Output exceeds the supported byte range.");
        return Result::ERROR;
    }

    U64 iterationWidth = 0;
    U64 candidateMaxIterations = 0;
    if (!detail::CheckedAdd(candidateSamplesPerSymbol, 4, iterationWidth) ||
        !detail::CheckedMultiply(outputSampleSize,
                                  iterationWidth,
                                  candidateMaxIterations) ||
        candidateMaxIterations == 0) {
        JST_ERROR("[MODULE_PSK_DEMOD] Iteration geometry exceeds the supported range.");
        return Result::ERROR;
    }

    const U64 candidateBatchSize = axes.batch ? inputTensor.shape(*axes.batch) : 1;
    U64 candidateOutputSymbolsPerLane = 0;
    U64 candidateMaxRecoveredSymbolsPerLane = 0;
    U64 candidatePendingSymbolsCapacity = 0;
    if (!detail::CheckedMultiply(outputSampleSize,
                                 candidateBatchSize,
                                 candidateOutputSymbolsPerLane) ||
        !detail::CheckedMultiply(candidateMaxIterations,
                                 candidateBatchSize,
                                 candidateMaxRecoveredSymbolsPerLane) ||
        !detail::CheckedAdd(candidateOutputSymbolsPerLane,
                            candidateMaxRecoveredSymbolsPerLane,
                            candidatePendingSymbolsCapacity) ||
        candidatePendingSymbolsCapacity >
            static_cast<U64>(std::numeric_limits<std::size_t>::max())) {
        JST_ERROR("[MODULE_PSK_DEMOD] Pending symbol capacity exceeds the supported range.");
        return Result::ERROR;
    }

    validatedOutputSize = candidateOutputSize;
    validatedOutputSizeBytes = candidateOutputSizeBytes;
    validatedMaxIterations = candidateMaxIterations;
    validatedInputSampleSize = inputSampleSize;
    validatedOutputSampleSize = outputSampleSize;
    validatedSampleHistoryCapacity = candidateSampleHistoryCapacity;
    validatedOutputSymbolsPerLane = candidateOutputSymbolsPerLane;
    validatedPendingSymbolsCapacity = candidatePendingSymbolsCapacity;
    validatedBatchSize = candidateBatchSize;
    validatedLaneCount = candidateLaneCount;
    validatedSignalAxes = axes;
    validatedOutputShape = std::move(candidateOutputShape);

    return Result::SUCCESS;
}

Result PskDemodImpl::define() {
    JST_CHECK(defineInterfaceInput("signal"));
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result PskDemodImpl::create() {
    const Tensor& inputTensor = inputs().at("signal").tensor;
    input = inputTensor;

    samplesPerSymbol = validatedSamplesPerSymbol;
    constellationOrder = validatedConstellationOrder;
    outputSize = validatedOutputSize;
    maxIterations = validatedMaxIterations;
    inputSampleSize = validatedInputSampleSize;
    outputSampleSize = validatedOutputSampleSize;
    sampleHistoryCapacity = validatedSampleHistoryCapacity;
    outputSymbolsPerLane = validatedOutputSymbolsPerLane;
    pendingSymbolsCapacity = validatedPendingSymbolsCapacity;
    batchSize = validatedBatchSize;
    laneCount = validatedLaneCount;
    sampleAxis = *validatedSignalAxes.sample;
    batchAxis = validatedSignalAxes.batch;
    freqAlpha = validatedFreqAlpha;
    freqBeta = validatedFreqBeta;
    timingAlpha = validatedTimingAlpha;
    timingBeta = validatedTimingBeta;
    timingOmegaNominal = validatedTimingOmegaNominal;
    timingOmegaMin = validatedTimingOmegaMin;
    timingOmegaMax = validatedTimingOmegaMax;

    // Allocate output tensor.
    JST_CHECK(output.create(input.device(), DataType::CF32, validatedOutputShape));
    JST_CHECK(output.propagateAttributes(input));
    JST_CHECK(SetSignalAxes(output, validatedSignalAxes));
    JST_CHECK(output.setAttribute("sampleRate", validatedOutputSampleRate));

    laneAxes.clear();
    for (Index axis = 0; axis < input.rank(); ++axis) {
        if (axis != sampleAxis && (!batchAxis || axis != *batchAxis)) {
            laneAxes.push_back(axis);
        }
    }

    laneStates.resize(laneCount);
    for (auto& state : laneStates) {
        JST_CHECK(initializeState(state));
    }
    frequencyError = 0.0;

    outputs()["signal"].produced(name(), "signal", output);

    return Result::SUCCESS;
}

Result PskDemodImpl::reconfigure() {
    const auto& config = *candidate();

    // Check if only loop parameters changed.
    if (config.pskType == pskType &&
        config.sampleRate == sampleRate &&
        config.symbolRate == symbolRate) {
        // Only loop parameters changed, update coefficients without recreation.
        frequencyLoopBandwidth = config.frequencyLoopBandwidth;
        timingLoopBandwidth = config.timingLoopBandwidth;
        dampingFactor = config.dampingFactor;
        freqAlpha = validatedFreqAlpha;
        freqBeta = validatedFreqBeta;
        timingAlpha = validatedTimingAlpha;
        timingBeta = validatedTimingBeta;
        return Result::SUCCESS;
    }

    // Core parameters changed, need recreation.
    return Result::RECREATE;
}

Result PskDemodImpl::initializeState(DemodState& state) {
    state.phaseAccumulator = 0.0;
    state.frequencyError = 0.0;
    state.timingMu = 0.0;
    state.timingOmega = timingOmegaNominal;
    state.timingIndex = 0;
    state.hasLastSymbol = false;
    state.lastSymbol = CF32{0.0f, 0.0f};
    state.lastDecision = CF32{0.0f, 0.0f};
    JST_CHECK(state.sampleHistory.resize(sampleHistoryCapacity));
    return state.pendingSymbols.resize(pendingSymbolsCapacity);
}

CF32 PskDemodImpl::interpolate(const CF32& a, const CF32& b, F64 mu) const {
    mu = std::clamp(mu, 0.0, 1.0);
    const F32 frac = static_cast<F32>(mu);
    const F32 inv = 1.0f - frac;
    return a * inv + b * frac;
}

CF32 PskDemodImpl::decision(const CF32& sample) const {
    constexpr F32 INV_SQRT2 = 0.7071067811865475f;
    constexpr F32 STEP_8PSK = kPi / 4.0f;

    switch (constellationOrder) {
        case 2: {
            // BPSK maps to the real axis.
            const F32 sign = sample.real() >= 0.0f ? 1.0f : -1.0f;
            return CF32(sign, 0.0f);
        }
        case 4: {
            // QPSK aligns to the quadrants.
            const F32 re = sample.real() >= 0.0f ? INV_SQRT2 : -INV_SQRT2;
            const F32 im = sample.imag() >= 0.0f ? INV_SQRT2 : -INV_SQRT2;
            return CF32(re, im);
        }
        case 8: {
            // 8-PSK selects the nearest constellation point.
            const F32 phase = std::arg(sample);
            const F32 decisionPhase = std::round(phase / STEP_8PSK) * STEP_8PSK;
            return std::polar(1.0f, decisionPhase);
        }
        default:
            return sample;
    }
}

F64 PskDemodImpl::muellerMullerError(const CF32& prevSymbol, const CF32& prevDecision,
                                     const CF32& currentSymbol, const CF32& currentDecision) const {
    const CF32 term1 = prevDecision * std::conj(currentSymbol);
    const CF32 term2 = prevSymbol * std::conj(currentDecision);
    return static_cast<F64>(std::real(term1 - term2));
}

F64 PskDemodImpl::costasLoopError(const CF32& sample) const {
    F64 error = 0.0;

    switch (constellationOrder) {
        case 2: {
            // BPSK.
            error = sample.imag() * (sample.real() > 0.0f ? 1.0f : -1.0f);
            break;
        }
        case 4: {
            // QPSK.
            const F32 reSign = sample.real() > 0.0f ? 1.0f : -1.0f;
            const F32 imSign = sample.imag() > 0.0f ? 1.0f : -1.0f;
            const CF32 decisionPoint(reSign, imSign);
            error = std::imag(sample * std::conj(decisionPoint));
            break;
        }
        case 8: {
            // 8-PSK.
            const F32 phase = std::arg(sample);
            const F32 decisionPhase = std::round(phase * 4.0f / kPi) * kPi / 4.0f;
            error = std::sin(phase - decisionPhase);
            break;
        }
        default:
            error = 0.0;
            break;
    }

    return std::clamp(error, MIN_FREQUENCY_ERROR, MAX_FREQUENCY_ERROR);
}

CF32 PskDemodImpl::correctFrequency(const CF32& sample, F64 phase) const {
    const CF32 correction = std::polar(1.0f, static_cast<F32>(-phase));
    return sample * correction;
}

}  // namespace Jetstream::Modules
