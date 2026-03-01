#include "module_impl.hh"

#include <algorithm>
#include <cmath>

namespace Jetstream::Modules {

Result PskDemodImpl::validate() {
    const auto& config = *candidate();

    if (config.sampleRate <= 0.0) {
        JST_ERROR("[MODULE_PSK_DEMOD] Sample rate must be positive.");
        return Result::ERROR;
    }

    if (config.symbolRate <= 0.0) {
        JST_ERROR("[MODULE_PSK_DEMOD] Symbol rate must be positive.");
        return Result::ERROR;
    }

    if (config.symbolRate >= config.sampleRate) {
        JST_ERROR("[MODULE_PSK_DEMOD] Symbol rate must be less than sample rate.");
        return Result::ERROR;
    }

    if (config.frequencyLoopBandwidth <= 0.0 || config.frequencyLoopBandwidth >= 1.0) {
        JST_ERROR("[MODULE_PSK_DEMOD] Frequency loop bandwidth must be between 0 and 1.");
        return Result::ERROR;
    }

    if (config.timingLoopBandwidth <= 0.0 || config.timingLoopBandwidth >= 1.0) {
        JST_ERROR("[MODULE_PSK_DEMOD] Timing loop bandwidth must be between 0 and 1.");
        return Result::ERROR;
    }

    if (config.dampingFactor <= 0.0) {
        JST_ERROR("[MODULE_PSK_DEMOD] Damping factor must be positive.");
        return Result::ERROR;
    }

    if (config.pskType != "bpsk" && config.pskType != "qpsk" && config.pskType != "8psk") {
        JST_ERROR("[MODULE_PSK_DEMOD] Unsupported PSK type: {}.", config.pskType);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result PskDemodImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS));

    JST_CHECK(defineInterfaceInput("signal"));
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result PskDemodImpl::create() {
    const Tensor& inputTensor = inputs().at("signal").tensor;
    input = inputTensor;

    // Calculate samples per symbol.
    samplesPerSymbol = static_cast<U64>(sampleRate / symbolRate);
    if (samplesPerSymbol < 2) {
        JST_ERROR("[MODULE_PSK_DEMOD] Samples per symbol must be at least 2.");
        return Result::ERROR;
    }

    // Calculate output size.
    const U64 inputSamples = input.size();
    outputSize = inputSamples / samplesPerSymbol;
    if (outputSize == 0) {
        JST_ERROR("[MODULE_PSK_DEMOD] Input buffer too small to produce any symbols.");
        return Result::ERROR;
    }

    // Determine constellation order from PSK type.
    if (pskType == "bpsk") {
        constellationOrder = 2;
    } else if (pskType == "qpsk") {
        constellationOrder = 4;
    } else if (pskType == "8psk") {
        constellationOrder = 8;
    }

    // Calculate output shape.
    std::vector<U64> outputShape = input.shape();
    outputShape[outputShape.size() - 1] = outputSize;

    // Allocate output tensor.
    JST_CHECK(output.create(input.device(), DataType::CF32, outputShape));
    JST_CHECK(output.propagateAttributes(input));

    // Initialize timing parameters.
    const F64 nominalOmega = sampleRate / symbolRate;
    timingOmegaNominal = nominalOmega;
    timingOmega = nominalOmega;
    timingOmegaMin = std::max(0.5, nominalOmega * 0.5);
    timingOmegaMax = std::max(timingOmegaMin + 1e-6, nominalOmega * 1.5);

    // Update loop filter coefficients.
    updateLoopCoefficients();

    // Initialize state.
    initializeState();

    outputs()["signal"] = {name(), "signal", output};

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
        updateLoopCoefficients();
        return Result::SUCCESS;
    }

    // Core parameters changed, need recreation.
    return Result::RECREATE;
}

void PskDemodImpl::updateLoopCoefficients() {
    const F64 damp = dampingFactor;

    // Frequency loop coefficients.
    F64 bw = frequencyLoopBandwidth;
    F64 denominator = 1.0 + 2.0 * damp * bw + bw * bw;
    freqAlpha = (4.0 * damp * bw) / denominator;
    freqBeta = (4.0 * bw * bw) / denominator;

    // Timing loop coefficients.
    bw = timingLoopBandwidth;
    denominator = 1.0 + 2.0 * damp * bw + bw * bw;
    timingAlpha = (4.0 * damp * bw) / denominator;
    timingBeta = (4.0 * bw * bw) / denominator;
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
    constexpr F32 PI_F = 3.14159265358979323846f;
    constexpr F32 STEP_8PSK = PI_F / 4.0f;

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
            constexpr F32 PI_F = 3.14159265358979323846f;
            const F32 decisionPhase = std::round(phase * 4.0f / PI_F) * PI_F / 4.0f;
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
