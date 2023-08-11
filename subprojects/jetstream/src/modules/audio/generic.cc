#include "jetstream/modules/audio.hh"

namespace Jetstream { 

template<Device D, typename T>
Audio<D, T>::Audio(const Config& config, 
                   const Input& input) 
         : config(config),
           input(input),
           buffer(input.buffer.shape()[1]*20) {
    JST_DEBUG("Initializing Audio module.");
    
    // Initialize input/output.
    JST_CHECK_THROW(Module::initInput(this->input.buffer));

    // Configure audio device.
    deviceConfig = ma_device_config_init(ma_device_type_playback);
    // TODO: Implement support for more audio formats.
    deviceConfig.playback.format   = ma_format_f32;
    // TODO: Implement support for more channels.
    deviceConfig.playback.channels = 1;
    deviceConfig.sampleRate        = static_cast<U32>(config.sampleRate);
    deviceConfig.dataCallback      = this->callback;
    deviceConfig.pUserData         = this;

    if (ma_device_init(nullptr, &deviceConfig, &deviceCtx) != MA_SUCCESS) {
        JST_FATAL("Failed to open audio device.");
        JST_CHECK_THROW(Result::ERROR);
    }

    if (ma_device_start(&deviceCtx) != MA_SUCCESS) {
        JST_FATAL("Failed to start playback device.");
        ma_device_uninit(&deviceCtx);
        JST_CHECK_THROW(Result::ERROR);
    }
}

template<Device D, typename T>
Audio<D, T>::~Audio() {
    ma_device_uninit(&deviceCtx);
}

template<Device D, typename T>
void Audio<D, T>::summary() const {
    JST_INFO("  Device Name:        {}", deviceCtx.playback.name);
    JST_INFO("  Sample Rate:        {:.2f} kHz", config.sampleRate / 1000);
}

template<Device D, typename T>
void Audio<D, T>::callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    auto* audio = reinterpret_cast<Audio<D, T>*>(pDevice->pUserData);

    if (frameCount < audio->buffer.getOccupancy()) {
        audio->buffer.get((float*)pOutput, frameCount);
    }
}

inline F32 phase(const CF32& c) {
    return atan2(c.imag(), c.real());
}

template<Device D, typename T>
inline std::vector<F32> fmDemodulate(const Vector<D, T, 2>& iq) {
    std::vector<F32> demodulated;
    demodulated.reserve(iq.size());
    double prevPhase = phase(iq[0]);

    for (size_t i = 1; i < iq.size(); ++i) {
        double currPhase = phase(iq[i]);
        double phaseDiff = currPhase - prevPhase;

        // Adjust phase difference to be between -pi and pi
        while (phaseDiff > M_PI) phaseDiff -= 2 * M_PI;
        while (phaseDiff < -M_PI) phaseDiff += 2 * M_PI;

        demodulated.push_back(phaseDiff);
        prevPhase = currPhase;
    }

    return demodulated;
}

inline std::vector<F32> resample(const std::vector<F32>& signal, int upFactor, int downFactor) {
    std::vector<F32> upsampled;
    upsampled.reserve(signal.size());
    std::vector<F32> filtered;
    filtered.reserve(signal.size());
    std::vector<F32> downsampled;
    downsampled.reserve(signal.size());

    // Upsampling
    for (double val : signal) {
        upsampled.push_back(val);
        for (int i = 1; i < upFactor; ++i) {
            upsampled.push_back(0.0);
        }
    }

    // Filtering (using a simple boxcar filter)
    int filterSize = upFactor;
    for (size_t i = 0; i < upsampled.size() - filterSize; ++i) {
        double sum = 0.0;
        for (int j = 0; j < filterSize; ++j) {
            sum += upsampled[i + j];
        }
        filtered.push_back(sum / filterSize);
    }

    // Downsampling
    for (size_t i = 0; i < filtered.size(); i += downFactor) {
        downsampled.push_back(filtered[i]);
    }

    return downsampled;
}

template<Device D, typename T>
Result Audio<D, T>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create Audio compute core.");
    return Result::SUCCESS;
}

template<Device D, typename T>
Result Audio<D, T>::compute(const RuntimeMetadata&) {
    const auto& demodulated = fmDemodulate(input.buffer);
    const auto& resampled = resample(demodulated, 1, 5);

    buffer.put(resampled.data(), resampled.size());

    //buffer.put((F32*)input.buffer.data(), (U64)input.buffer.shape()[1]/5);

    return Result::SUCCESS;
}

}  // namespace Jetstream
