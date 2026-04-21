#include "module_impl.hh"

#include <cmath>
#include <numeric>

#include "miniaudio.h"

#ifdef JST_OS_BROWSER
#include <emscripten.h>
#endif

namespace Jetstream::Modules {

AudioImpl::AudioImpl() = default;
AudioImpl::~AudioImpl() = default;

struct AudioImpl::Impl {
    ma_device_config deviceConfig;
    ma_device deviceCtx;
    ma_resampler_config resamplerConfig;
    ma_resampler resamplerCtx;

    static void callback(ma_device* pDevice, void* pOutput, const void* pInput,
                         ma_uint32 frameCount);
    static std::vector<std::pair<ma_device_id, std::string>> GetAvailableDevices();
    static void GenerateUniqueName(std::string& name, const ma_device_id& id);
};

void AudioImpl::Impl::GenerateUniqueName(std::string& name, const ma_device_id& id) {
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
        const U64 sum = std::accumulate(id.wasapi, id.wasapi + sizeof(id.wasapi), 0ULL);
        name = jst::fmt::format("{} ({:08X})", name, sum);
    } else if (id.dsound[0] != '\0') {
        const U64 sum = std::accumulate(id.dsound, id.dsound + sizeof(id.dsound), 0ULL);
        name = jst::fmt::format("{} ({:08X})", name, sum);
    }
}

std::vector<std::pair<ma_device_id, std::string>> AudioImpl::Impl::GetAvailableDevices() {
    std::vector<std::pair<ma_device_id, std::string>> devices;

    devices.push_back({{}, "Default"});

    ma_context context;

    if (ma_context_init(NULL, 0, NULL, &context) != MA_SUCCESS) {
        JST_ERROR("[MODULE_AUDIO] Failed to initialize audio context.");
        return devices;
    }

    ma_device_info* pPlaybackDeviceInfos;
    ma_uint32 playbackDeviceCount;

    if (ma_context_get_devices(&context, &pPlaybackDeviceInfos, &playbackDeviceCount,
                               nullptr, nullptr) != MA_SUCCESS) {
        JST_ERROR("[MODULE_AUDIO] Failed to retrieve audio devices.");
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

void AudioImpl::Impl::callback(ma_device* pDevice, void* pOutput, const void*,
                               ma_uint32 frameCount) {
    auto* audioCircularBuffer = reinterpret_cast<Tools::CircularBuffer<F32>*>(pDevice->pUserData);

    if (frameCount < audioCircularBuffer->getOccupancy()) {
        audioCircularBuffer->get(reinterpret_cast<F32*>(pOutput), frameCount);
    }
}

AudioImpl::DeviceList AudioImpl::ListAvailableDevices() {
    const auto& devices = Impl::GetAvailableDevices();

    DeviceList deviceList;
    for (const auto& [_, name] : devices) {
        deviceList.push_back(name);
    }

    return deviceList;
}

Result AudioImpl::validate() {
    const auto& config = *candidate();

    if (config.inSampleRate <= 0) {
        JST_ERROR("[MODULE_AUDIO] Input sample rate must be positive.");
        return Result::ERROR;
    }

    if (config.outSampleRate <= 0) {
        JST_ERROR("[MODULE_AUDIO] Output sample rate must be positive.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result AudioImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::BROWSER_MAIN_THREAD));

    JST_CHECK(defineInterfaceInput("buffer"));

    return Result::SUCCESS;
}

Result AudioImpl::create() {
    pimpl = std::make_unique<Impl>();

    const auto& inputBuffer = inputs().at("buffer").tensor;

    // Configure audio resampler.

    pimpl->resamplerConfig = ma_resampler_config_init(
        ma_format_f32,
        1,
        static_cast<U32>(inSampleRate),
        static_cast<U32>(outSampleRate),
        ma_resample_algorithm_linear
    );
    pimpl->resamplerConfig.linear.lpfOrder = 8;

    if (ma_resampler_init(&pimpl->resamplerConfig, nullptr, &pimpl->resamplerCtx) != MA_SUCCESS) {
        JST_ERROR("[MODULE_AUDIO] Failed to create audio resampler.");
        return Result::ERROR;
    }

    // Get available audio devices.

    const auto& devices = Impl::GetAvailableDevices();

    if (devices.empty()) {
        JST_ERROR("[MODULE_AUDIO] No audio devices found.");
        return Result::INCOMPLETE;
    }

    ma_device_id selectedDeviceId;
    bool foundConfigDevice = false;
    bool useDefaultDevice = deviceName == "Default" ||
                            deviceName == "default" ||
                            deviceName.empty();

    JST_DEBUG("[MODULE_AUDIO] Found audio devices:");
    for (U64 i = 0; i < devices.size(); i++) {
        const auto& id = devices[i].first;
        std::string name = devices[i].second;

        if (name == deviceName) {
            selectedDeviceId = id;
            foundConfigDevice = true;
        }

        JST_DEBUG("[MODULE_AUDIO]   [{}]: {}", i, name);
    }

    if (!foundConfigDevice && !useDefaultDevice) {
        JST_WARN("[MODULE_AUDIO] Device '{}' not found, using default.", deviceName);
    }

    // Configure audio device.

    pimpl->deviceConfig = ma_device_config_init(ma_device_type_playback);
    pimpl->deviceConfig.playback.pDeviceID = (!foundConfigDevice || useDefaultDevice) ?
                                              nullptr : &selectedDeviceId;
    pimpl->deviceConfig.playback.format = ma_format_f32;
    pimpl->deviceConfig.playback.channels = 1;
    pimpl->deviceConfig.sampleRate = static_cast<U32>(outSampleRate);
    pimpl->deviceConfig.dataCallback = Impl::callback;
    pimpl->deviceConfig.pUserData = &circularBuffer;

    if (ma_device_init(nullptr, &pimpl->deviceConfig, &pimpl->deviceCtx) != MA_SUCCESS) {
        JST_ERROR("[MODULE_AUDIO] Failed to open audio device.");
        ma_resampler_uninit(&pimpl->resamplerCtx, nullptr);
        return Result::INCOMPLETE;
    }

    resolvedDeviceName = pimpl->deviceCtx.playback.name;

    if (ma_device_start(&pimpl->deviceCtx) != MA_SUCCESS) {
        JST_ERROR("[MODULE_AUDIO] Failed to start playback device.");
        ma_device_uninit(&pimpl->deviceCtx);
        ma_resampler_uninit(&pimpl->resamplerCtx, nullptr);
        return Result::ERROR;
    }

    // Set initial volume.
    ma_device_set_master_volume(&pimpl->deviceCtx, volume);

    // Allocate resampler scratch buffer.

    const U64 outputSize = static_cast<U64>(inputBuffer.size() * (outSampleRate / inSampleRate));
    JST_CHECK(buffer.create(device(), DataType::F32, {outputSize}));

    // Initialize circular buffer.

    circularBuffer.resize(inputBuffer.size() * 20);

    return Result::SUCCESS;
}

Result AudioImpl::destroy() {
    if (pimpl) {
        ma_device_uninit(&pimpl->deviceCtx);
        ma_resampler_uninit(&pimpl->resamplerCtx, nullptr);
        pimpl.reset();
    }

    return Result::SUCCESS;
}

Result AudioImpl::reconfigure() {
    const auto& config = *candidate();
    constexpr F32 EPSILON = 1e-6f;

    if (config.deviceName != deviceName ||
        std::abs(config.inSampleRate - inSampleRate) > EPSILON ||
        std::abs(config.outSampleRate - outSampleRate) > EPSILON) {
        return Result::RECREATE;
    }

    if (std::abs(config.volume - volume) > EPSILON) {
        volume = config.volume;
        if (pimpl) {
            ma_device_set_master_volume(&pimpl->deviceCtx, volume);
        }
    }

    return Result::SUCCESS;
}

const std::string& AudioImpl::getDeviceName() const {
    return resolvedDeviceName;
}

Result AudioImpl::resample() {
    const auto& input = inputs().at("buffer").tensor;

    ma_uint64 frameCountIn = input.size();
    ma_uint64 frameCountOut = buffer.size();

    ma_result result = ma_resampler_process_pcm_frames(
        &pimpl->resamplerCtx,
        input.data(),
        &frameCountIn,
        buffer.data(),
        &frameCountOut
    );

    if (result != MA_SUCCESS) {
        JST_ERROR("[MODULE_AUDIO] Failed to resample audio signal.");
        return Result::ERROR;
    }

    circularBuffer.put(reinterpret_cast<F32*>(buffer.data()), frameCountOut);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
