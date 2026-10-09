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
    Result reconfigure() override;

 protected:
    std::shared_ptr<Modules::Soapy> moduleConfig = std::make_shared<Modules::Soapy>();
    Modules::SoapyImpl* moduleImpl = nullptr;

 private:
    Result updateAntennaFormat();
    Parser::Map antennaFormat(const std::string& selected,
                              const std::vector<std::string>& antennas = {});
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

    moduleConfig->modulePath = modulePath;
    moduleConfig->deviceString = deviceString;
    moduleConfig->streamString = streamString;
    moduleConfig->antenna = antenna;
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

    Parser::Sequence deviceOptions{Parser::Map{{"label", "None"}, {"value", ""}}};
    bool selectionListed = config.deviceString.empty();
    for (const auto& [label, device] : Modules::SoapyDiscovery::ListDevices()) {
        auto args = device;
        args.erase("label");
        const auto value = SoapySDR::KwargsToString(args);
        selectionListed = selectionListed || value == config.deviceString;
        deviceOptions.emplace_back(Parser::Map{{"label", label}, {"value", value}});
    }
    if (!selectionListed) {
        deviceOptions.emplace_back(Parser::Map{{"label", "Configured device"}, {"value", config.deviceString}});
    }

    JST_CHECK(defineInterfaceConfig("deviceString",
                                    "Device",
                                    "Select a device to receive samples. Choose None to disconnect.",
                                    {{"type", "dropdown"}, {"options", std::move(deviceOptions)}}));

    JST_CHECK(defineInterfaceConfig("antenna",
                                    "Antenna",
                                    "Receive antenna port. Changes restart the receiver.",
                                    antennaFormat(config.antenna)));

    JST_CHECK(defineInterfaceConfig("frequency",
                                    "Frequency",
                                    "Tuner frequency.",
                                    {{"type", "float"}, {"unit", "MHz"}, {"scale", 1.0e6f},
                                     {"precision", 3}, {"step_config",
                                        std::isfinite(config.frequencyStep) && config.frequencyStep > 0.0f
                                            ? std::string("frequencyStep") : std::string{}}}));

    JST_CHECK(defineInterfaceConfig("sampleRate",
                                    "Sample Rate",
                                    "Sampling rate.",
                                    {{"type", "float"}, {"unit", "MHz"}, {"scale", 1.0e6f}, {"precision", 3}}));

    JST_CHECK(defineInterfaceConfig("automaticGain",
                                    "Automatic Gain",
                                    "Enable automatic gain control.",
                                    {{"type", "bool"}}));

    JST_CHECK(defineInterfaceConfig("biasTee",
                                    "Bias-T",
                                    "Enable antenna power when supported by the selected device.",
                                    {{"type", "bool"}}));

    JST_CHECK(defineInterfaceConfig("numberOfBatches",
                                    "Batches",
                                    "Number of batches in output buffer.",
                                    {{"type", "uint"}, {"unit", "batches"}}));

    JST_CHECK(defineInterfaceConfig("numberOfTimeSamples",
                                    "Samples",
                                    "Samples per batch and the FFT size downstream.",
                                    {{"type", "uint"}, {"unit", "samples"}}));

    JST_CHECK(defineInterfaceConfig("bufferMultiplier",
                                    "Buffer Multiplier",
                                    "Internal buffer size multiplier.",
                                    {{"type", "uint"}, {"unit", "x"}}));

    JST_CHECK(defineInterfaceMetric("bufferHealth",
                                    "Buffer Health",
                                    "Current buffer occupancy level.",
                                    {{"type", "progressbar"}},
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
                                    "Buffer loss percentage and reported device overflows (OVF) since stream start.",
                                    {{"type", "progressbar"}},
        [this]() -> std::any {
            const F64 loss = moduleImpl ? moduleImpl->getBufferLoss() : 0.0;
            const U64 deviceOverflows = moduleImpl ? moduleImpl->getDeviceOverflows() : 0;
            auto label = loss > 0.0 && loss < 0.0001
                ? std::string("<0.01%") : jst::fmt::format("{:.2f}%", loss * 100.0);
            if (deviceOverflows > 0) {
                label = jst::fmt::format("{} ({} OVF)", label, deviceOverflows);
            }
            return std::pair<std::string, F32>{label, static_cast<F32>(loss)};
        }));

    JST_CHECK(defineInterfaceMetric("throughput",
                                    "Throughput",
                                    "Current data throughput.",
                                    {{"type", "label"}},
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
    JST_CHECK(updateAntennaFormat());

    return Result::SUCCESS;
}

Result SoapyImpl::reconfigure() {
    JST_CHECK(updateAntennaFormat());

    return Result::SUCCESS;
}

Result SoapyImpl::updateAntennaFormat() {
    return updateInterfaceConfigFormat("antenna",
                                       antennaFormat(antenna, moduleImpl->listAntennas()));
}

Parser::Map SoapyImpl::antennaFormat(const std::string& selected,
                                     const std::vector<std::string>& antennas) {
    Parser::Sequence options{Parser::Map{{"label", "Default"}, {"value", ""}}};
    bool selectionListed = selected.empty();
    for (const auto& antenna : antennas) {
        selectionListed = selectionListed || antenna == selected;
        options.emplace_back(Parser::Map{{"label", antenna}, {"value", antenna}});
    }
    if (!selectionListed) {
        options.emplace_back(Parser::Map{{"label", selected}, {"value", selected}});
    }
    return {{"type", "dropdown"}, {"options", std::move(options)}};
}

JST_REGISTER_BLOCK(SoapyImpl, {"soapy"});

}  // namespace Jetstream::Blocks
