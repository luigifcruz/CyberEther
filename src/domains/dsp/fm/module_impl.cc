#include "module_impl.hh"

#include <cmath>

namespace Jetstream::Modules {

Result FmImpl::validate() {
    validatedSignalAxes = {};
    validatedLaneCount = 0;

    const auto& config = *candidate();

    if (config.mode != "narrow" && config.mode != "wide") {
        JST_ERROR("[MODULE_FM] Mode must be 'narrow' or 'wide'.");
        return Result::ERROR;
    }

    if (config.deemphasis != "none" && config.deemphasis != "50us" &&
        config.deemphasis != "75us") {
        JST_ERROR("[MODULE_FM] De-emphasis must be 'none', '50us', or '75us'.");
        return Result::ERROR;
    }

    if (!std::isfinite(config.sampleRate) || config.sampleRate <= 0.0f) {
        JST_ERROR("[MODULE_FM] Sample rate must be finite and positive.");
        return Result::ERROR;
    }

    constexpr F32 maxSampleRate = 20e6f;
    if (config.sampleRate > maxSampleRate) {
        JST_ERROR("[MODULE_FM] Sample rate must not exceed 20 MHz.");
        return Result::ERROR;
    }

    if (config.mode == "wide" && config.sampleRate < 200e3f) {
        JST_ERROR("[MODULE_FM] Wideband mode requires a sample rate of at "
                  "least 200 kHz.");
        return Result::ERROR;
    }

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (ResolveSignalAxes(inputTensor, validatedSignalAxes) != Result::SUCCESS) {
        JST_ERROR("[MODULE_FM] Input must contain valid signal axis metadata.");
        return Result::ERROR;
    }

    if (config.mode == "wide" && validatedSignalAxes.channel) {
        JST_ERROR("[MODULE_FM] Wideband mode does not support channelized input.");
        return Result::ERROR;
    }

    validatedLaneCount = inputTensor.size() /
                         inputTensor.shape(*validatedSignalAxes.sample);
    if (validatedSignalAxes.batch) {
        validatedLaneCount /= inputTensor.shape(*validatedSignalAxes.batch);
    }

    return Result::SUCCESS;
}

Result FmImpl::define() {
    JST_CHECK(defineInterfaceInput("signal"));
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result FmImpl::create() {
    const Tensor& inputTensor = inputs().at("signal").tensor;
    input = inputTensor;
    signalAxes = validatedSignalAxes;
    laneCount = validatedLaneCount;
    wideBand = mode == "wide";
    deemphasisEnabled = deemphasis != "none";

    // Initialize coefficients.
    updateCoefficients();
    previousSample.assign(laneCount, CF32{0.0f, 0.0f});
    hasPreviousSample.assign(laneCount, U8{0});
    narrowDeemphasisState.assign(wideBand ? 0 : laneCount, 0.0f);
    stereoState.assign(wideBand ? laneCount : 0, StereoState{});

    // Allocate output tensor (real F32).
    Shape outputShape = input.shape();
    SignalAxes outputAxes = signalAxes;
    if (wideBand) {
        outputAxes.channel = outputShape.size();
        outputShape.push_back(2);
    }
    JST_CHECK(output.create(input.device(), DataType::F32, outputShape));
    JST_CHECK(output.propagateAttributes(input));
    JST_CHECK(SetSignalAxes(output, outputAxes));
    JST_CHECK(output.setAttribute("frequency", F32{0.0f}));

    outputs()["signal"].produced(name(), "signal", output);

    return Result::SUCCESS;
}

void FmImpl::updateCoefficients() {
    const F32 deviation = wideBand ? 75e3f : 100e3f;
    kf = deviation / sampleRate;
    ref = 1.0f / (2.0f * JST_PI * kf);

    pilotPhaseIncrement = 2.0f * JST_PI * 19e3f / sampleRate;
    const F64 sampleRateF64 = sampleRate;
    pilotAlpha = static_cast<F32>(
        1.0 - std::exp(-2.0 * JST_PI * 200.0 / sampleRateF64));
    if (deemphasis == "none") {
        deemphasisAlpha = 1.0f;
    } else {
        const F64 deemphasisTime = deemphasis == "50us" ? 50e-6 : 75e-6;
        deemphasisAlpha = static_cast<F32>(
            1.0 - std::exp(-1.0 / (sampleRateF64 * deemphasisTime)));
    }

    const F64 pilotOmega = 2.0 * JST_PI * 19e3 / sampleRateF64;
    const F64 pilotCosine = std::cos(pilotOmega);
    const F64 pilotSine = std::sin(pilotOmega);
    const F64 pilotNotchAlpha = pilotSine / (2.0 * 20.0);
    const F64 pilotNotchA0 = 1.0 + pilotNotchAlpha;
    pilotNotch.b0 = static_cast<F32>(1.0 / pilotNotchA0);
    pilotNotch.b1 = static_cast<F32>(-2.0 * pilotCosine / pilotNotchA0);
    pilotNotch.b2 = pilotNotch.b0;
    pilotNotch.a1 = pilotNotch.b1;
    pilotNotch.a2 = static_cast<F32>(
        (1.0 - pilotNotchAlpha) / pilotNotchA0);

    constexpr std::array<F64, 3> q = {
        0.51763809,
        0.70710678,
        1.93185165,
    };
    const F64 omega = 2.0 * JST_PI * 15e3 / sampleRateF64;
    const F64 cosine = std::cos(omega);
    const F64 sine = std::sin(omega);
    for (U64 section = 0; section < audioLowPass.size(); ++section) {
        const F64 alpha = sine / (2.0 * q[section]);
        const F64 a0 = 1.0 + alpha;
        auto& coefficients = audioLowPass[section];
        coefficients.b0 = static_cast<F32>((1.0 - cosine) * 0.5 / a0);
        coefficients.b1 = static_cast<F32>((1.0 - cosine) / a0);
        coefficients.b2 = coefficients.b0;
        coefficients.a1 = static_cast<F32>(-2.0 * cosine / a0);
        coefficients.a2 = static_cast<F32>((1.0 - alpha) / a0);
    }
}

F32 FmImpl::applyBiquad(F32 sample, const Biquad& coefficients,
                         BiquadState& state) const {
    const F32 outputSample = coefficients.b0 * sample + state.z1;
    state.z1 = coefficients.b1 * sample -
               coefficients.a1 * outputSample + state.z2;
    state.z2 = coefficients.b2 * sample - coefficients.a2 * outputSample;
    return outputSample;
}

F32 FmImpl::applyAudioLowPass(F32 sample,
                              std::array<BiquadState, 3>& state) const {
    for (U64 section = 0; section < audioLowPass.size(); ++section) {
        const auto& coefficients = audioLowPass[section];
        sample = applyBiquad(sample, coefficients, state[section]);
    }
    return sample;
}

}  // namespace Jetstream::Modules
