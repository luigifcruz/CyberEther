#include "module_impl.hh"

#include <cmath>
#include <limits>

#include <jetstream/memory/axis.hh>
#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

Result SignalGeneratorImpl::validate() {
    const auto& config = *candidate();
    validatedDataType = DataType::None;
    validatedOutputSizeBytes = 0;
    constexpr F64 minF32 = static_cast<F64>(std::numeric_limits<F32>::denorm_min());
    constexpr F64 maxF32 = static_cast<F64>(std::numeric_limits<F32>::max());

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

    DataType dataType = DataType::None;
    if (config.signalDataType == "F32") {
        dataType = DataType::F32;
    } else if (config.signalDataType == "CF32") {
        dataType = DataType::CF32;
    } else {
        JST_ERROR("[MODULE_SIGNAL_GENERATOR] Invalid data type '{}'.", config.signalDataType);
        return Result::ERROR;
    }

    const bool isSinusoid = config.signalType == "sine" ||
                            config.signalType == "cosine";
    const bool isPeriodic = isSinusoid || config.signalType == "square" ||
                            config.signalType == "triangle" ||
                            config.signalType == "sawtooth";
    const bool isNoise = config.signalType == "noise";
    const bool isDc = config.signalType == "dc";
    const bool isChirp = config.signalType == "chirp";
    const bool isComplex = dataType == DataType::CF32;

    if (!std::isfinite(config.sampleRate) ||
        config.sampleRate < minF32 || config.sampleRate > maxF32) {
        JST_ERROR("[MODULE_SIGNAL_GENERATOR] Sample rate must be positive and "
                  "within the F32 range ({}).", config.sampleRate);
        return Result::ERROR;
    }

    const F64 nyquist = config.sampleRate * 0.5;
    if (isPeriodic) {
        const F64 minimumFrequency = isComplex && isSinusoid ? -nyquist : 0.0;
        if (!std::isfinite(config.frequency) ||
            config.frequency < minimumFrequency || config.frequency > nyquist) {
            JST_ERROR("[MODULE_SIGNAL_GENERATOR] Frequency ({}) must be within "
                      "the supported range [{}, {}].", config.frequency,
                      minimumFrequency, nyquist);
            return Result::ERROR;
        }
    }

    if (!std::isfinite(config.amplitude) ||
        config.amplitude < 0.0 || config.amplitude > maxF32) {
        JST_ERROR("[MODULE_SIGNAL_GENERATOR] Amplitude must be non-negative "
                  "and within the F32 range ({}).", config.amplitude);
        return Result::ERROR;
    }

    if ((isPeriodic || isChirp) && !std::isfinite(config.phase)) {
        JST_ERROR("[MODULE_SIGNAL_GENERATOR] Phase must be finite ({}).",
                  config.phase);
        return Result::ERROR;
    }

    if (!std::isfinite(config.dcOffset) || std::abs(config.dcOffset) > maxF32) {
        JST_ERROR("[MODULE_SIGNAL_GENERATOR] DC offset must be within the F32 "
                  "range ({}).", config.dcOffset);
        return Result::ERROR;
    }

    if (isDc) {
        const F64 dcValue = config.amplitude + config.dcOffset;
        if (!std::isfinite(dcValue) || std::abs(dcValue) > maxF32) {
            JST_ERROR("[MODULE_SIGNAL_GENERATOR] DC value exceeds the F32 "
                      "output range.");
            return Result::ERROR;
        }
    } else if (!isNoise &&
               config.amplitude > maxF32 - std::abs(config.dcOffset)) {
        JST_ERROR("[MODULE_SIGNAL_GENERATOR] Amplitude and DC offset exceed "
                  "the F32 output range.");
        return Result::ERROR;
    }

    if (config.bufferSize == 0) {
        JST_ERROR("[MODULE_SIGNAL_GENERATOR] Buffer size cannot be zero.");
        return Result::ERROR;
    }

    if (isNoise) {
        if (!std::isfinite(config.noiseVariance) ||
            config.noiseVariance < 0.0 || config.noiseVariance > maxF32) {
            JST_ERROR("[MODULE_SIGNAL_GENERATOR] Noise variance must be "
                      "non-negative and within the F32 range ({}).",
                      config.noiseVariance);
            return Result::ERROR;
        }
    }

    if (isChirp) {
        const F64 minimumFrequency = isComplex ? -nyquist : 0.0;
        if (!std::isfinite(config.chirpStartFreq) ||
            config.chirpStartFreq < minimumFrequency ||
            config.chirpStartFreq > nyquist) {
            JST_ERROR("[MODULE_SIGNAL_GENERATOR] Chirp start frequency ({}) "
                      "must be within [{}, {}].", config.chirpStartFreq,
                      minimumFrequency, nyquist);
            return Result::ERROR;
        }

        if (!std::isfinite(config.chirpEndFreq) ||
            config.chirpEndFreq < minimumFrequency ||
            config.chirpEndFreq > nyquist) {
            JST_ERROR("[MODULE_SIGNAL_GENERATOR] Chirp end frequency ({}) "
                      "must be within [{}, {}].", config.chirpEndFreq,
                      minimumFrequency, nyquist);
            return Result::ERROR;
        }

        const F64 minimumDuration = 1.0 / config.sampleRate;
        if (!std::isfinite(config.chirpDuration) ||
            config.chirpDuration < minimumDuration ||
            config.chirpDuration > maxF32) {
            JST_ERROR("[MODULE_SIGNAL_GENERATOR] Chirp duration ({}) must be "
                      "at least one sample period ({}).", config.chirpDuration,
                      minimumDuration);
            return Result::ERROR;
        }
    }

    U64 outputSizeBytes = 0;
    if (!detail::CheckedMultiply(config.bufferSize,
                                 static_cast<U64>(DataTypeSize(dataType)),
                                 outputSizeBytes)) {
        JST_ERROR("[MODULE_SIGNAL_GENERATOR] Output byte size exceeds the supported range.");
        return Result::ERROR;
    }

    validatedDataType = dataType;
    validatedOutputSizeBytes = outputSizeBytes;
    return Result::SUCCESS;
}

Result SignalGeneratorImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result SignalGeneratorImpl::create() {
    JST_CHECK(signal.create(device(), validatedDataType, {bufferSize}));
    JST_CHECK(SetSignalAxes(signal, {
        .sample = Index{0},
    }));

    JST_CHECK(signal.setAttribute("frequency", F32{0.0f}));
    JST_CHECK(signal.setAttribute("sampleRate", static_cast<F32>(sampleRate)));

    outputs()["signal"].produced(name(), "signal", signal);

    return Result::SUCCESS;
}

Result SignalGeneratorImpl::destroy() {
    return Result::SUCCESS;
}

Result SignalGeneratorImpl::reconfigure() {
    const auto& config = *candidate();
    if (config.signalType != signalType ||
        config.signalDataType != signalDataType ||
        config.sampleRate != sampleRate ||
        config.bufferSize != bufferSize) {
        return Result::RECREATE;
    }

    signalType = config.signalType;
    signalDataType = config.signalDataType;
    sampleRate = config.sampleRate;
    frequency = config.frequency;
    amplitude = config.amplitude;
    phase = config.phase;
    dcOffset = config.dcOffset;
    noiseVariance = config.noiseVariance;
    chirpStartFreq = config.chirpStartFreq;
    chirpEndFreq = config.chirpEndFreq;
    chirpDuration = config.chirpDuration;
    bufferSize = config.bufferSize;

    JST_CHECK(signal.setAttribute("frequency", F32{0.0f}));
    JST_CHECK(signal.setAttribute("sampleRate", static_cast<F32>(sampleRate)));

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
