#include "jetstream/modules/audio.hh"

namespace Jetstream {

template<Device D, typename T>
Result Audio<D, T>::create() {
    JST_DEBUG("Initializing Audio module.");

    // Initialize input/output.
    JST_INIT(
        JST_INIT_INPUT("buffer", input.buffer);
    );

    // Initialize circular buffer.
    buffer.resize(input.buffer.shape()[1]*20);

    // Configure audio resampler.
    resamplerConfig = ma_resampler_config_init(
        ma_format_f32,
        1, 
        static_cast<U32>(config.inSampleRate),
        static_cast<U32>(config.outSampleRate),
        ma_resample_algorithm_linear
    );
    resamplerConfig.linear.lpfOrder = 8;

    tmp.resize( input.buffer.size() * (config.outSampleRate / config.inSampleRate));

    if (ma_resampler_init(&resamplerConfig, nullptr, &resamplerCtx) != MA_SUCCESS) {
        JST_ERROR("Failed to create audio resampler.");
        JST_CHECK(Result::ERROR);
    }

    // Configure audio device.
    deviceConfig = ma_device_config_init(ma_device_type_playback);
    // TODO: Implement support for more audio formats.
    deviceConfig.playback.format   = ma_format_f32;
    // TODO: Implement support for more channels.
    deviceConfig.playback.channels = 1;
    deviceConfig.sampleRate        = static_cast<U32>(config.outSampleRate);
    deviceConfig.dataCallback      = callback;
    deviceConfig.pUserData         = this;

    if (ma_device_init(nullptr, &deviceConfig, &deviceCtx) != MA_SUCCESS) {
        JST_ERROR("Failed to open audio device.");
        JST_CHECK(Result::ERROR);
    }

    if (ma_device_start(&deviceCtx) != MA_SUCCESS) {
        JST_ERROR("Failed to start playback device.");
        ma_device_uninit(&deviceCtx);
        JST_CHECK(Result::ERROR);
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Audio<D, T>::destroy() {
    JST_TRACE("Audio killed.");

    ma_resampler_uninit(&resamplerCtx, nullptr);
    ma_device_uninit(&deviceCtx);

    return Result::SUCCESS;
}

template<Device D, typename T>
void Audio<D, T>::summary() const {
    JST_INFO("  Device Name:        {}", deviceCtx.playback.name);
    JST_INFO("  Input Sample Rate:  {:.2f} kHz", config.inSampleRate / 1000);
    JST_INFO("  Output Sample Rate: {:.2f} kHz", config.outSampleRate / 1000);
}

template<Device D, typename T>
void Audio<D, T>::callback(ma_device* pDevice, void* pOutput, const void*, ma_uint32 frameCount) {
    auto* audio = reinterpret_cast<Audio<D, T>*>(pDevice->pUserData);

    if (frameCount < audio->buffer.getOccupancy()) {
        audio->buffer.get(reinterpret_cast<F32*>(pOutput), frameCount);
    }
}

template<Device D, typename T>
Result Audio<D, T>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create Audio compute core.");
    return Result::SUCCESS;
}

template<Device D, typename T>
Result Audio<D, T>::compute(const RuntimeMetadata&) {
    ma_uint64 frameCountIn  = input.buffer.size();
    ma_uint64 frameCountOut = tmp.size();

    // TODO: Remove copy.
    ma_result result = ma_resampler_process_pcm_frames(&resamplerCtx, input.buffer.data(), &frameCountIn, tmp.data(), &frameCountOut);
    if (result != MA_SUCCESS) {
        JST_ERROR("Failed to resample signal.");
        return Result::ERROR;
    }

    buffer.put(tmp.data(), frameCountOut);

    return Result::SUCCESS;
}

}  // namespace Jetstream
