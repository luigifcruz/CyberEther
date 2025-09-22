#include <jetstream/domains/io/audio/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/io/audio/module.hh>

#include "miniaudio.h"

namespace Jetstream::Blocks {

static std::vector<std::string> ListAvailableDevices() {
    std::vector<std::string> devices;

    devices.push_back("Default");

    ma_context context;

    if (ma_context_init(NULL, 0, NULL, &context) != MA_SUCCESS) {
        JST_ERROR("[BLOCK_AUDIO] Failed to initialize audio context.");
        return devices;
    }

    ma_device_info* pPlaybackDeviceInfos;
    ma_uint32 playbackDeviceCount;

    if (ma_context_get_devices(&context, &pPlaybackDeviceInfos, &playbackDeviceCount,
                               nullptr, nullptr) != MA_SUCCESS) {
        JST_ERROR("[BLOCK_AUDIO] Failed to retrieve audio devices.");
        ma_context_uninit(&context);
        return devices;
    }

    for (ma_uint32 i = 0; i < playbackDeviceCount; i++) {
        devices.push_back(pPlaybackDeviceInfos[i].name);
    }

    ma_context_uninit(&context);

    return devices;
}

struct AudioImpl : public Block::Impl, public DynamicConfig<Blocks::Audio> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Audio> moduleConfig = std::make_shared<Modules::Audio>();
    std::string deviceDropdown;
};

Result AudioImpl::configure() {
    moduleConfig->deviceName = deviceName;
    moduleConfig->inSampleRate = inSampleRate;
    moduleConfig->outSampleRate = outSampleRate;

    return Result::SUCCESS;
}

Result AudioImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer",
                                   "Input",
                                   "The input buffer containing audio samples to play."));

    std::vector<std::string> deviceOptions;
    for (const auto& device : ListAvailableDevices()) {
        deviceOptions.push_back(jst::fmt::format("{}({})", device, device));
    }
    deviceDropdown = jst::fmt::format("dropdown:{}", jst::fmt::join(deviceOptions, ","));

    JST_CHECK(defineInterfaceConfig("deviceName",
                                    "Device",
                                    "Select from available audio devices.",
                                    deviceDropdown));

    JST_CHECK(defineInterfaceConfig("inSampleRate",
                                    "Sample Rate",
                                    "Sample rate of the input signal.",
                                    "float:kHz:1"));

    return Result::SUCCESS;
}

Result AudioImpl::create() {
    JST_CHECK(moduleCreate("audio", moduleConfig, {
        {"buffer", inputs().at("buffer")}
    }));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(AudioImpl);

}  // namespace Jetstream::Blocks
