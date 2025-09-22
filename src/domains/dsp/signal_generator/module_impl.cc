#include "module_impl.hh"

namespace Jetstream::Modules {

Result SignalGeneratorImpl::validate() {
    const auto& config = *candidate();

    if (config.signalType != "sine" &&
        config.signalType != "cosine" &&
        config.signalType != "square" &&
        config.signalType != "triangle" &&
        config.signalType != "sawtooth" &&
        config.signalType != "noise" &&
        config.signalType != "dc" &&
        config.signalType != "chirp") {
        JST_ERROR("[MODULE_SIGNAL_GENERATOR] Invalid signal type '{}'.", config.signalType);
        return Result::ERROR;
    }

    if (config.signalDataType != "F32" &&
        config.signalDataType != "CF32") {
        JST_ERROR("[MODULE_SIGNAL_GENERATOR] Invalid data type '{}'.", config.signalDataType);
        return Result::ERROR;
    }

    if (config.sampleRate <= 0.0f) {
        JST_ERROR("[MODULE_SIGNAL_GENERATOR] Sample rate must be positive ({}).", config.sampleRate);
        return Result::ERROR;
    }

    if (config.frequency < 0.0f) {
        JST_ERROR("[MODULE_SIGNAL_GENERATOR] Frequency cannot be negative ({}).", config.frequency);
        return Result::ERROR;
    }

    if (config.amplitude < 0.0f) {
        JST_ERROR("[MODULE_SIGNAL_GENERATOR] Amplitude cannot be negative ({}).", config.amplitude);
        return Result::ERROR;
    }

    if (config.bufferSize == 0) {
        JST_ERROR("[MODULE_SIGNAL_GENERATOR] Buffer size cannot be zero.");
        return Result::ERROR;
    }

    if (config.noiseVariance < 0.0f) {
        JST_ERROR("[MODULE_SIGNAL_GENERATOR] Noise variance cannot be negative ({}).", config.noiseVariance);
        return Result::ERROR;
    }

    if (config.signalType == "chirp") {
        if (config.chirpStartFreq < 0.0f) {
            JST_ERROR("[MODULE_SIGNAL_GENERATOR] Chirp start frequency cannot be negative ({}).", config.chirpStartFreq);
            return Result::ERROR;
        }

        if (config.chirpEndFreq < 0.0f) {
            JST_ERROR("[MODULE_SIGNAL_GENERATOR] Chirp end frequency cannot be negative ({}).", config.chirpEndFreq);
            return Result::ERROR;
        }

        if (config.chirpDuration <= 0.0f) {
            JST_ERROR("[MODULE_SIGNAL_GENERATOR] Chirp duration must be positive ({}).", config.chirpDuration);
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

Result SignalGeneratorImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result SignalGeneratorImpl::create() {
    JST_CHECK(signal.create(device(), NameToDataType(signalDataType), {bufferSize}));

    signal.setAttribute("frequency", static_cast<F32>(frequency));
    signal.setAttribute("sampleRate", static_cast<F32>(sampleRate));

    outputs()["signal"] = {name(), "signal", signal};

    return Result::SUCCESS;
}

Result SignalGeneratorImpl::destroy() {
    return Result::SUCCESS;
}

Result SignalGeneratorImpl::reconfigure() {
    // TODO: Implement update logic for SignalGeneratorImpl.
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
