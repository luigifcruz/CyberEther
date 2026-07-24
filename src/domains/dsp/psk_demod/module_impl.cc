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
    validatedOutputShape.clear();
    validatedFreqAlpha = 0.0;
    validatedFreqBeta = 0.0;
    validatedTimingAlpha = 0.0;
    validatedTimingBeta = 0.0;
    validatedTimingOmegaNominal = 0.0;
    validatedTimingOmegaMin = 0.0;
    validatedTimingOmegaMax = 0.0;

    const auto& config = *candidate();

    if (!std::isfinite(config.sampleRate) || config.sampleRate <= 0.0) {
        JST_ERROR("[MODULE_PSK_DEMOD] Sample rate must be finite and positive.");
        return Result::ERROR;
    }

    if (!std::isfinite(config.symbolRate) || config.symbolRate <= 0.0) {
        JST_ERROR("[MODULE_PSK_DEMOD] Symbol rate must be finite and positive.");
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

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.rank() != 1) {
        JST_ERROR("[MODULE_PSK_DEMOD] Input must be a rank-one tensor.");
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

    const U64 candidateOutputSize = inputTensor.size() / candidateSamplesPerSymbol;
    if (candidateOutputSize == 0) {
        JST_ERROR("[MODULE_PSK_DEMOD] Input buffer too small to produce any symbols.");
        return Result::ERROR;
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
        !detail::CheckedMultiply(candidateOutputSize,
                                 iterationWidth,
                                 candidateMaxIterations) ||
        candidateMaxIterations == 0) {
        JST_ERROR("[MODULE_PSK_DEMOD] Iteration geometry exceeds the supported range.");
        return Result::ERROR;
    }

    validatedOutputSize = candidateOutputSize;
    validatedOutputSizeBytes = candidateOutputSizeBytes;
    validatedMaxIterations = candidateMaxIterations;
    validatedOutputShape = {candidateOutputSize};

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

    // Initialize state.
    initializeState();

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

void PskDemodImpl::initializeState() {
    phaseAccumulator = 0.0;
    frequencyError = 0.0;
    timingMu = 0.0;
    timingOmega = timingOmegaNominal;
    timingIndex = 0;
    hasLastSymbol = false;
    lastSymbol = CF32{0.0f, 0.0f};
    lastDecision = CF32{0.0f, 0.0f};
    sampleHistory.clear();
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
