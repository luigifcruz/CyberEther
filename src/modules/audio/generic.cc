#include "jetstream/modules/audio.hh"

#include "miniaudio.h"

namespace Jetstream {

template<Device D, typename T>
Audio<D, T>::Audio() {
    pimpl = std::make_unique<Impl>();
}

template<Device D, typename T>
Audio<D, T>::~Audio() {
    pimpl.reset();
}

template<Device D, typename T>
struct Audio<D, T>::Impl {
    ma_device_config deviceConfig;
    ma_device deviceCtx;
    ma_resampler_config resamplerConfig;
    ma_resampler resamplerCtx;

    Memory::CircularBuffer<F32> buffer;  

    static void callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount);
    static std::vector<std::pair<ma_device_id, std::string>> GetAvailableDevice();
    static void GenerateUniqueName(std::string& name, const ma_device_id& id);
};

template<Device D, typename T>
void Audio<D, T>::Impl::GenerateUniqueName(std::string& name, const ma_device_id& id) {
    if (id.pulse[0] != '\0') {
        name = jst::fmt::format("{} ({})", name, std::string_view(id.pulse));
    } else if (id.alsa[0] != '\0') {
        name = jst::fmt::format("{} ({})", name, std::string_view(id.alsa));
    } else if (id.jack != 0) {
        name = jst::fmt::format("{} ({})", name, id.jack);
    } else if (id.coreaudio[0] != '\0') {
        name = jst::fmt::format("{} ({})", name, std::string_view(id.coreaudio));
    } else if (id.sndio[0] != '\0') {
        name = jst::fmt::format("{} ({})", name, std::string_view(id.sndio));
    } else if (id.audio4[0] != '\0') {
        name = jst::fmt::format("{} ({})", name, std::string_view(id.audio4));
    } else if (id.oss[0] != '\0') {
        name = jst::fmt::format("{} ({})", name, std::string_view(id.oss));
    } else if (id.aaudio != 0) {
        name = jst::fmt::format("{} ({})", name, id.aaudio);
    } else if (id.opensl != 0) {
        name = jst::fmt::format("{} ({})", name, id.opensl);
    } else if (id.webaudio[0] != '\0') {
        name = jst::fmt::format("{} ({})", name, std::string_view(id.webaudio));
    } else if (id.custom.i != 0) {
        name = jst::fmt::format("{} ({})", name, id.custom.i);
    } else if (id.nullbackend != 0) {
        name = jst::fmt::format("{} ({})", name, id.nullbackend);
    } else if (id.winmm != 0) {
        name = jst::fmt::format("{} ({})", name, id.winmm);
    } else if (id.wasapi[0] != '\0') {
        // TODO: Implement wchar to string conversion.
        const U64 sum = std::accumulate(id.wasapi, id.wasapi + sizeof(id.wasapi), 0);
        name = jst::fmt::format("{} ({:08X})", name, sum);
    } else if (id.dsound[0] != '\0') {
        // TODO: Implement GUID to string conversion.
        const U64 sum = std::accumulate(id.dsound, id.dsound + sizeof(id.dsound), 0);
        name = jst::fmt::format("{} ({:08X})", name, sum);
    }
}

template<Device D, typename T>
std::vector<std::pair<ma_device_id, std::string>> Audio<D, T>::Impl::GetAvailableDevice() {
    std::vector<std::pair<ma_device_id, std::string>> devices;

    devices.push_back({
        {}, "Default",
    });

    ma_context context;

    if (ma_context_init(NULL, 0, NULL, &context) != MA_SUCCESS) {
        JST_ERROR("Failed to initialize context.\n");
        return devices;
    }

    ma_device_info* pPlaybackDeviceInfos;
    ma_uint32 playbackDeviceCount;

    if (ma_context_get_devices(&context, &pPlaybackDeviceInfos, &playbackDeviceCount, nullptr, nullptr) != MA_SUCCESS) {
        JST_ERROR("Failed to retrieve audio devices.");
        ma_context_uninit(&context);
        return devices;
    }

    std::unordered_map<std::string, U64> nameCount;

    for (ma_uint32 i = 0; i < playbackDeviceCount; i++) {
        nameCount[pPlaybackDeviceInfos[i].name] = 0;
    }

    for (ma_uint32 i = 0; i < playbackDeviceCount; i++) {
        nameCount[pPlaybackDeviceInfos[i].name] += 1;
    }

    for (ma_uint32 i = 0; i < playbackDeviceCount; i++) {
        const auto& id = pPlaybackDeviceInfos[i].id;
        std::string name = pPlaybackDeviceInfos[i].name;

        if (nameCount.at(name) > 1) {
            Impl::GenerateUniqueName(name, id);
        }

        devices.push_back({id, name});
    }

    ma_context_uninit(&context);

    return devices;
}

template<Device D, typename T>
typename Audio<D, T>::DeviceList Audio<D, T>::ListAvailableDevices() {
    const auto& devices = Impl::GetAvailableDevice();

    DeviceList deviceList;
    for (const auto& [_, name] : devices) {
        deviceList.push_back(name);
    }

    return deviceList;
}

template<Device D, typename T>
Result Audio<D, T>::create() {
    JST_DEBUG("Initializing Audio module.");
    JST_INIT_IO();

    // Configure audio resampler.

    pimpl->resamplerConfig = ma_resampler_config_init(
        ma_format_f32,
        1, 
        static_cast<U32>(config.inSampleRate),
        static_cast<U32>(config.outSampleRate),
        ma_resample_algorithm_linear
    );
    pimpl->resamplerConfig.linear.lpfOrder = 8;

    if (ma_resampler_init(&pimpl->resamplerConfig, nullptr, &pimpl->resamplerCtx) != MA_SUCCESS) {
        JST_ERROR("Failed to create audio resampler.");
        return Result::ERROR;
    }

    // Get available audio devices.
    
    const auto& devices = Impl::GetAvailableDevice();
    
    if (devices.empty()) {
        JST_ERROR("No audio devices found.");
        return Result::ERROR;
    }

    ma_device_id selectedDeviceId;
    bool foundConfigDevice = false;
    bool useDefaultDevice = config.deviceName == "Default" || 
                            config.deviceName == "default" || 
                            config.deviceName == "";

    JST_DEBUG("Found audio device:");
    for (U64 i = 0; i < devices.size(); i++) {
        const auto& id = devices[i].first;
        std::string name = devices[i].second;

        if (name == config.deviceName) {
            selectedDeviceId = id;
            foundConfigDevice = true;
        }

        JST_DEBUG("    [{}]: {}", i, name);
    }

    if (!foundConfigDevice && !useDefaultDevice) {
        JST_WARN("No audio device with name '{}' found. Using default device.", config.deviceName);
    }

    // Configure audio device.

    pimpl->deviceConfig = ma_device_config_init(ma_device_type_playback);
    pimpl->deviceConfig.playback.pDeviceID = (!foundConfigDevice || useDefaultDevice) ? nullptr : &selectedDeviceId;
    pimpl->deviceConfig.playback.format    = ma_format_f32;
    pimpl->deviceConfig.playback.channels  = 1;  // TODO: Support for more channels.
    pimpl->deviceConfig.sampleRate         = static_cast<U32>(config.outSampleRate);
    pimpl->deviceConfig.dataCallback       = Impl::callback;
    pimpl->deviceConfig.pUserData          = pimpl.get();

    if (ma_device_init(nullptr, &pimpl->deviceConfig, &pimpl->deviceCtx) != MA_SUCCESS) {
        JST_ERROR("Failed to open audio device.");
        return Result::ERROR;
    }

    deviceName = pimpl->deviceCtx.playback.name;

    if (ma_device_start(&pimpl->deviceCtx) != MA_SUCCESS) {
        JST_ERROR("Failed to start playback device.");
        ma_device_uninit(&pimpl->deviceCtx);
        return Result::ERROR;
    }

    // Allocate output.

    const U64 outputSize = input.buffer.size() * (config.outSampleRate / config.inSampleRate);
    output.buffer = Tensor<D, T>({outputSize});

    // Initialize circular buffer.

    pimpl->buffer.resize(input.buffer.shape()[1]*20);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Audio<D, T>::destroy() {
    JST_TRACE("Audio killed.");

    ma_resampler_uninit(&pimpl->resamplerCtx, nullptr);
    ma_device_uninit(&pimpl->deviceCtx);

    return Result::SUCCESS;
}

template<Device D, typename T>
void Audio<D, T>::info() const {
    JST_INFO("  Device Name:        {}", config.deviceName);
    JST_INFO("  Input Sample Rate:  {:.2f} kHz", config.inSampleRate / 1000);
    JST_INFO("  Output Sample Rate: {:.2f} kHz", config.outSampleRate / 1000);
}

template<Device D, typename T>
void Audio<D, T>::Impl::callback(ma_device* pDevice, void* pOutput, const void*, ma_uint32 frameCount) {
    auto* audio = reinterpret_cast<Audio<D, T>::Impl*>(pDevice->pUserData);

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
    ma_uint64 frameCountOut = output.buffer.size();

    // TODO: Create standalone resampler module.
    ma_result result = ma_resampler_process_pcm_frames(&pimpl->resamplerCtx, input.buffer.data(), &frameCountIn, output.buffer.data(), &frameCountOut);
    if (result != MA_SUCCESS) {
        JST_ERROR("Failed to resample signal.");
        return Result::ERROR;
    }

    JST_ASSERT(frameCountOut == output.buffer.size());

    pimpl->buffer.put(output.buffer.data(), frameCountOut);

    return Result::SUCCESS;
}

JST_AUDIO_CPU(JST_INSTANTIATION)

}  // namespace Jetstream
