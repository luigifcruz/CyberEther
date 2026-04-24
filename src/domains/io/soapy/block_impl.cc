#include <jetstream/domains/io/soapy/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/io/soapy/module.hh>
#include "module_impl.hh"

#include <SoapySDR/Device.hpp>
#include <SoapySDR/Types.hpp>

namespace Jetstream::Blocks {

using DeviceEntry = std::map<std::string, std::string>;
using DeviceList = std::map<std::string, DeviceEntry>;

static DeviceList ListAvailableDevices(const std::string& filter) {
    DeviceList deviceMap;
    const SoapySDR::Kwargs args = SoapySDR::KwargsFromString(filter);

    try {
        for (const auto& device : SoapySDR::Device::enumerate(args)) {
            deviceMap[device.at("label")] = device;
        }
    } catch (const std::exception& e) {
        JST_ERROR("[BLOCK_SOAPY] Failed to enumerate devices: {}", e.what());
    } catch (...) {
        JST_ERROR("[BLOCK_SOAPY] Failed to enumerate devices.");
    }

    return deviceMap;
}

struct SoapyImpl : public Block::Impl, public DynamicConfig<Blocks::Soapy> {
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Soapy> moduleConfig = std::make_shared<Modules::Soapy>();
    Modules::SoapyImpl* moduleImpl = nullptr;
    std::string deviceDropdown;
};

Result SoapyImpl::configure() {
    std::string resolvedDeviceString;
    const auto availableDeviceList = ListAvailableDevices(hintString);
    const auto selectFirstAvailable = [&](const DeviceList& devices) -> bool {
        if (devices.empty()) {
            return false;
        }

        const auto& [label, device] = *devices.begin();
        deviceString = label;
        resolvedDeviceString = SoapySDR::KwargsToString(device);
        return true;
    };

    if (const auto it = availableDeviceList.find(deviceString); it != availableDeviceList.end()) {
        resolvedDeviceString = SoapySDR::KwargsToString(it->second);
    } else if (!deviceString.empty()) {
        const auto explicitDeviceList = ListAvailableDevices(deviceString);
        if (!selectFirstAvailable(explicitDeviceList)) {
            selectFirstAvailable(availableDeviceList);
        }
    } else if (!availableDeviceList.empty()) {
        selectFirstAvailable(availableDeviceList);
    }

    moduleConfig->deviceString = resolvedDeviceString;
    moduleConfig->streamString = streamString;
    moduleConfig->frequency = frequency;
    moduleConfig->sampleRate = sampleRate;
    moduleConfig->automaticGain = automaticGain;
    moduleConfig->numberOfBatches = numberOfBatches;
    moduleConfig->numberOfTimeSamples = numberOfTimeSamples;
    moduleConfig->bufferMultiplier = bufferMultiplier;

    return Result::SUCCESS;
}

Result SoapyImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "The output buffer containing samples from the SDR device."));

    std::vector<std::string> deviceOptions;
    for (const auto& [label, _] : ListAvailableDevices(hintString)) {
        deviceOptions.push_back(jst::fmt::format("{}({})", label, label));
    }
    deviceDropdown = jst::fmt::format("dropdown:{}", jst::fmt::join(deviceOptions, ","));

    JST_CHECK(defineInterfaceConfig("deviceString",
                                    "Device",
                                    "Select from available SDR devices.",
                                    deviceDropdown));

    JST_CHECK(defineInterfaceConfig("hintString",
                                    "Device Hint",
                                    "Filter string for discovering devices.",
                                    "text"));

    JST_CHECK(defineInterfaceConfig("frequency",
                                    "Frequency",
                                    "Tuner frequency.",
                                    "float:MHz:3:frequencyStep"));

    JST_CHECK(defineInterfaceConfig("sampleRate",
                                    "Sample Rate",
                                    "Sampling rate.",
                                    "float:MHz:3"));

    JST_CHECK(defineInterfaceConfig("automaticGain",
                                    "Automatic Gain",
                                    "Enable automatic gain control.",
                                    "bool"));

    JST_CHECK(defineInterfaceConfig("numberOfBatches",
                                    "Batches",
                                    "Number of batches in output buffer.",
                                    "int:batches"));

    JST_CHECK(defineInterfaceConfig("numberOfTimeSamples",
                                    "Samples",
                                    "Number of samples per batch.",
                                    "int:samples"));

    JST_CHECK(defineInterfaceConfig("bufferMultiplier",
                                    "Buffer Multiplier",
                                    "Internal buffer size multiplier.",
                                    "int:x"));

    JST_CHECK(defineInterfaceMetric("bufferHealth",
                                    "Buffer Health",
                                    "Current buffer occupancy level.",
                                    "progressbar",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::pair<std::string, F32>{"0.0%", 0.0f};
            }
            const F32 bufferHealth = moduleImpl->getBufferHealth();
            return std::pair<std::string, F32>{jst::fmt::format("{:.1f}%", bufferHealth * 100.0f),
                                               bufferHealth};
        }));

    JST_CHECK(defineInterfaceMetric("throughput",
                                    "Throughput",
                                    "Current data throughput.",
                                    "label",
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::string("N/A");
            }
            const auto [actual, expected] = moduleImpl->getThroughput();
            return jst::fmt::format("{:.1f} / {:.1f} MB/s", actual, expected);
        }));

    return Result::SUCCESS;
}

Result SoapyImpl::create() {
    JST_CHECK(moduleCreate("soapy", moduleConfig, {}));
    JST_CHECK(moduleExposeOutput("signal", {"soapy", "signal"}));

    moduleImpl = moduleHandle("soapy")->getImpl<Modules::SoapyImpl>();

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(SoapyImpl);

}  // namespace Jetstream::Blocks
