#include <jetstream/domains/io/soapy/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/io/soapy/module.hh>
#include "module_impl.hh"
#include "soapysdr.hh"

#include <SoapySDR/Types.hpp>

#include <cmath>

namespace Jetstream::Blocks {

struct SoapyImpl : public Block::Impl, public DynamicConfig<Blocks::Soapy> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Soapy> moduleConfig = std::make_shared<Modules::Soapy>();
    Modules::SoapyImpl* moduleImpl = nullptr;
    std::string deviceDropdown;
};

Result SoapyImpl::validate() {
    const auto& config = *candidate();

    if (!std::isfinite(config.frequencyStep) || config.frequencyStep <= 0.0f) {
        JST_ERROR("[BLOCK_SOAPY] Frequency step must be finite and positive.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result SoapyImpl::configure() {
    JST_CHECK(Modules::SoapyDiscovery::LoadDriverLibrary(modulePath));

    std::string resolvedDeviceString;
    const auto availableDeviceList = Modules::SoapyDiscovery::ListDevices(hintString);
    const auto selectFirstAvailable = [&](const Modules::SoapyDiscovery::DeviceList& devices) -> bool {
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
        const auto explicitDeviceList = Modules::SoapyDiscovery::ListDevices(deviceString);
        if (!selectFirstAvailable(explicitDeviceList)) {
            selectFirstAvailable(availableDeviceList);
        }
    } else if (!availableDeviceList.empty()) {
        selectFirstAvailable(availableDeviceList);
    }

    moduleConfig->modulePath = modulePath;
    moduleConfig->deviceString = resolvedDeviceString;
    moduleConfig->streamString = streamString;
    moduleConfig->frequency = frequency;
    moduleConfig->sampleRate = sampleRate;
    moduleConfig->automaticGain = automaticGain;
    moduleConfig->biasTee = biasTee;
    moduleConfig->numberOfBatches = numberOfBatches;
    moduleConfig->numberOfTimeSamples = numberOfTimeSamples;
    moduleConfig->bufferMultiplier = bufferMultiplier;

    return Result::SUCCESS;
}

Result SoapyImpl::define() {
    const auto& config = *candidate();

    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "The output buffer containing samples from the SDR device."));

    std::vector<std::string> deviceOptions;
    for (const auto& [label, _] :
         Modules::SoapyDiscovery::ListDevices(config.hintString)) {
        deviceOptions.push_back(jst::fmt::format("{}({})", label, label));
    }
    deviceDropdown = jst::fmt::format("dropdown:{}", jst::fmt::join(deviceOptions, ","));

    JST_CHECK(defineInterfaceConfig("deviceString",
                                    "Device",
                                    "Select from available SDR devices.",
                                    deviceDropdown));

    JST_CHECK(defineInterfaceConfig("frequency",
                                    "Frequency",
                                    "Tuner frequency.",
                                    std::isfinite(config.frequencyStep) && config.frequencyStep > 0.0f
                                        ? "float:MHz:3:frequencyStep"
                                        : "float:MHz:3"));

    JST_CHECK(defineInterfaceConfig("sampleRate",
                                    "Sample Rate",
                                    "Sampling rate.",
                                    "float:MHz:3"));

    JST_CHECK(defineInterfaceConfig("automaticGain",
                                    "Automatic Gain",
                                    "Enable automatic gain control.",
                                    "bool"));

    JST_CHECK(defineInterfaceConfig("biasTee",
                                    "Bias-T",
                                    "Enable antenna power when supported by the selected device.",
                                    "bool"));

    JST_CHECK(defineInterfaceConfig("numberOfBatches",
                                    "Batches",
                                    "Number of batches in output buffer.",
                                    "uint:batches"));

    JST_CHECK(defineInterfaceConfig("numberOfTimeSamples",
                                    "Samples",
                                    "Number of samples per batch.",
                                    "uint:samples"));

    JST_CHECK(defineInterfaceConfig("bufferMultiplier",
                                    "Buffer Multiplier",
                                    "Internal buffer size multiplier.",
                                    "uint:x"));

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

    JST_CHECK(defineInterfaceMetric("bufferLoss",
                                    "Buffer Loss",
                                    "Percentage of received samples discarded because receive buffer was full. "
                                    "Cumulative since stream start. Excludes samples lost inside the device or driver. "
                                    "Resets when the stream is recreated.",
                                    "progressbar",
        [this]() -> std::any {
            const F64 loss = moduleImpl ? moduleImpl->getBufferLoss() : 0.0;
            const auto percentage = loss > 0.0 && loss < 0.0001
                ? std::string("<0.01%") : jst::fmt::format("{:.2f}%", loss * 100.0);
            return std::pair<std::string, F32>{percentage, static_cast<F32>(loss)};
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

JST_REGISTER_BLOCK(SoapyImpl, {"soapy"});

}  // namespace Jetstream::Blocks
