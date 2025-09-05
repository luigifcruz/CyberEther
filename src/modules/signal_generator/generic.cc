#include "jetstream/modules/signal_generator.hh"
#include "jetstream/render/macros.hh"

#include "benchmark.cc"

namespace Jetstream {

template<Device D, typename T>
Result SignalGenerator<D, T>::create() {
    JST_DEBUG("Initializing SignalGenerator module.");
    JST_INIT_IO();

    // Validate configuration
    if (config.bufferSize == 0) {
        JST_ERROR("Buffer size must be greater than 0.");
        return Result::ERROR;
    }

    if (config.sampleRate <= 0) {
        JST_ERROR("Sample rate must be greater than 0.");
        return Result::ERROR;
    }

    if (config.amplitude < 0) {
        JST_ERROR("Amplitude must be non-negative.");
        return Result::ERROR;
    }

    // Allocate output buffer
    if constexpr (D == Device::CUDA) {
        output.buffer = Tensor<D, T>({config.bufferSize}, true);
    } else {
        output.buffer = Tensor<D, T>({config.bufferSize});
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
void SignalGenerator<D, T>::info() const {
    JST_DEBUG("  Signal Type: {}", config.signalType);
    JST_DEBUG("  Sample Rate: {:.2f} MHz", config.sampleRate / JST_MHZ);
    JST_DEBUG("  Frequency: {:.3f} MHz", config.frequency / JST_MHZ);
    JST_DEBUG("  Amplitude: {}", config.amplitude);
    JST_DEBUG("  Phase: {} rad", config.phase);
    JST_DEBUG("  DC Offset: {}", config.dcOffset);
    JST_DEBUG("  Buffer Size: {}", config.bufferSize);

    if (config.signalType == SignalType::Noise) {
        JST_DEBUG("  Noise Variance: {}", config.noiseVariance);
    }

    if (config.signalType == SignalType::Chirp) {
        JST_DEBUG("  Chirp Start Frequency: {:.3f} MHz", config.chirpStartFreq / JST_MHZ);
        JST_DEBUG("  Chirp End Frequency: {:.3f} MHz", config.chirpEndFreq / JST_MHZ);
        JST_DEBUG("  Chirp Duration: {} s", config.chirpDuration);
    }
}

template<Device D, typename T>
Result SignalGenerator<D, T>::setFrequency(F64& frequency) {
    config.frequency = frequency;
    return Result::SUCCESS;
}

template<Device D, typename T>
Result SignalGenerator<D, T>::setSampleRate(F64& sampleRate) {
    if (sampleRate <= 0) {
        JST_WARN("Sample rate must be greater than 0.");
        sampleRate = config.sampleRate;
        return Result::WARNING;
    }

    config.sampleRate = sampleRate;
    return Result::SUCCESS;
}

template<Device D, typename T>
Result SignalGenerator<D, T>::setAmplitude(F64& amplitude) {
    if (amplitude < 0) {
        JST_WARN("Amplitude must be non-negative.");
        amplitude = config.amplitude;
        return Result::WARNING;
    }

    config.amplitude = amplitude;
    return Result::SUCCESS;
}

template<Device D, typename T>
Result SignalGenerator<D, T>::setPhase(F64& phase) {
    config.phase = phase;
    return Result::SUCCESS;
}

template<Device D, typename T>
Result SignalGenerator<D, T>::setDcOffset(F64& dcOffset) {
    config.dcOffset = dcOffset;
    return Result::SUCCESS;
}

template<Device D, typename T>
Result SignalGenerator<D, T>::setNoiseVariance(F64& noiseVariance) {
    if (noiseVariance <= 0) {
        JST_WARN("Noise variance must be greater than 0.");
        noiseVariance = config.noiseVariance;
        return Result::WARNING;
    }

    config.noiseVariance = noiseVariance;
    return Result::SUCCESS;
}

template<Device D, typename T>
Result SignalGenerator<D, T>::setChirpStartFreq(F64& chirpStartFreq) {
    config.chirpStartFreq = chirpStartFreq;
    return Result::SUCCESS;
}

template<Device D, typename T>
Result SignalGenerator<D, T>::setChirpEndFreq(F64& chirpEndFreq) {
    config.chirpEndFreq = chirpEndFreq;
    return Result::SUCCESS;
}

template<Device D, typename T>
Result SignalGenerator<D, T>::setChirpDuration(F64& chirpDuration) {
    if (chirpDuration <= 0) {
        JST_WARN("Chirp duration must be greater than 0.");
        chirpDuration = config.chirpDuration;
        return Result::WARNING;
    }

    config.chirpDuration = chirpDuration;
    return Result::SUCCESS;
}

}  // namespace Jetstream
