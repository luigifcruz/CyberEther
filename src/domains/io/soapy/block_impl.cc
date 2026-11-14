#include <jetstream/domains/io/soapy/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/io/soapy/module.hh>
#include "module_impl.hh"
#include "library/soapysdr.hh"

#include <cmath>
#include <limits>

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
    Result updateManualGainFormat();
    Parser::Map antennaFormat(const std::string& selected,
                              const std::vector<std::string>& antennas = {});
    Parser::Map manualGainFormat(const std::optional<Modules::SoapyAbi::Range>& range = std::nullopt);
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
    if (Modules::SoapyRuntime::Acquire()) {
        JST_CHECK(Modules::SoapyDiscovery::LoadDriverLibrary(modulePath));
    }

    moduleConfig->modulePath = modulePath;
    moduleConfig->deviceString = deviceString;
    moduleConfig->streamString = streamString;
    moduleConfig->antenna = antenna;
    moduleConfig->frequency = frequency;
    moduleConfig->sampleRate = sampleRate;
    moduleConfig->automaticGain = automaticGain;
    moduleConfig->manualGain = manualGain;
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
    for (const auto& [label, device] : Modules::SoapyDiscovery::List()) {
        auto args = device;
        args.erase("label");
        const auto value = Modules::SoapyDiscovery::FormatArguments(args);
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

    if (!config.automaticGain) {
        JST_CHECK(defineInterfaceConfig("manualGain",
                                        "Manual Gain",
                                        "Receive gain applied while automatic gain is off.",
                                        manualGainFormat()));
    }

    const auto module = moduleHandle("soapy");
    const bool knownWithoutBiasTee = module && module->state() == Module::State::CREATED &&
                                     config.deviceString == deviceString &&
                                     !module->getImpl<Modules::SoapyImpl>()->supportsBiasTee();
    if (!knownWithoutBiasTee) {
        JST_CHECK(defineInterfaceConfig("biasTee",
                                        "Bias-T",
                                        "Enable antenna power through the antenna cable.",
                                        {{"type", "bool"}}));
    }

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

    JST_CHECK(defineInterfaceMetric("bufferFill",
                                    "Buffer Fill",
                                    "How full the internal buffer is. Near 100% means the chain is falling behind.",
                                    {{"type", "progressbar"}, {"warning_above", 0.5f}, {"error_above", 0.75f}},
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::pair<std::string, F32>{"0.0%", 0.0f};
            }
            const F32 bufferFill = moduleImpl->getBufferFill();
            return std::pair<std::string, F32>{jst::fmt::format("{:.0f}%", bufferFill * 100.0f),
                                               bufferFill};
        }));

    JST_CHECK(defineInterfaceMetric("bufferLoss",
                                    "Buffer Loss",
                                    "Share of samples overwritten before the chain read them since the stream started. "
                                    "Overflows the radio reported (OVF) show next to it.",
                                    {{"type", "progressbar"}, {"warning_above", 0.0f}, {"error_above", 0.01f}},
        [this]() -> std::any {
            const F64 loss = moduleImpl ? moduleImpl->getBufferLoss() : 0.0;
            const U64 deviceOverflows = moduleImpl ? moduleImpl->getDeviceOverflows() : 0;
            auto label = loss > 0.0 && loss < 0.001
                ? std::string("<0.1%") : jst::fmt::format("{:.1f}%", loss * 100.0);
            if (deviceOverflows > 0) {
                label = jst::fmt::format("{} ({} OVF)", label, deviceOverflows);
            }
            return std::pair<std::string, F32>{label, static_cast<F32>(loss)};
        }));

    JST_CHECK(defineInterfaceMetric("throughput",
                                    "Throughput",
                                    "Rate the chain reads next to the rate the radio sends.",
                                    {{"type", "progressbar"}, {"warning_below", 0.95f}, {"error_below", 0.8f}},
        [this]() -> std::any {
            const auto [actual, expected] = moduleImpl ? moduleImpl->getThroughput()
                                                       : std::pair<F32, F32>{0.0f, 0.0f};
            const F32 fraction = actual > 0.0f && expected > 0.0f ? actual / expected
                                                                  : std::numeric_limits<F32>::quiet_NaN();
            return std::pair<std::string, F32>{
                jst::fmt::format("{:.1f} / {:.1f} MB/s", actual, expected), fraction};
        }));

    return Result::SUCCESS;
}

Result SoapyImpl::create() {
    JST_CHECK(moduleCreate("soapy", moduleConfig, {}));
    JST_CHECK(moduleExposeOutput("signal", {"soapy", "signal"}));

    moduleImpl = moduleHandle("soapy")->getImpl<Modules::SoapyImpl>();
    JST_CHECK(updateAntennaFormat());
    JST_CHECK(updateManualGainFormat());

    return Result::SUCCESS;
}

Result SoapyImpl::reconfigure() {
    JST_CHECK(updateAntennaFormat());
    JST_CHECK(updateManualGainFormat());

    return Result::SUCCESS;
}

Result SoapyImpl::updateAntennaFormat() {
    return updateInterfaceConfigFormat("antenna",
                                       antennaFormat(antenna, moduleImpl->listAntennas()));
}

Result SoapyImpl::updateManualGainFormat() {
    if (moduleImpl->getGainRange()) {
        manualGain = moduleImpl->manualGain;
        moduleConfig->manualGain = manualGain;
    }

    if (!automaticGain) {
        JST_CHECK(updateInterfaceConfigFormat("manualGain",
                                              manualGainFormat(moduleImpl->getGainRange())));
    }

    return Result::SUCCESS;
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

Parser::Map SoapyImpl::manualGainFormat(const std::optional<Modules::SoapyAbi::Range>& range) {
    const F32 minimum = range ? static_cast<F32>(range->minimum) : 0.0f;
    const F32 maximum = range ? static_cast<F32>(range->maximum) : 60.0f;
    return {{"type", "range"}, {"min", minimum}, {"max", maximum}, {"unit", "dB"}};
}

JST_REGISTER_BLOCK(SoapyImpl, {"soapy"});

}  // namespace Jetstream::Blocks
