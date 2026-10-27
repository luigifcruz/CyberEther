#include <jetstream/domains/io/sdr/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/io/sdr/module.hh>
#include "module_impl.hh"

#include <cmath>

namespace Jetstream::Blocks {

struct SdrImpl : public Block::Impl, public DynamicConfig<Blocks::Sdr> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;
    Result reconfigure() override;

 protected:
    std::shared_ptr<Modules::Sdr> moduleConfig = std::make_shared<Modules::Sdr>();
    Modules::SdrImpl* moduleImpl = nullptr;

 private:
    const Modules::SdrImpl* createdModule(const Blocks::Sdr& config);
    Parser::Map manualGainFormat(const Blocks::Sdr& config);
    Parser::Map antennaFormat(const Blocks::Sdr& config);
    Parser::Map sampleRateFormat(const Blocks::Sdr& config);
};

Result SdrImpl::validate() {
    const auto& config = *candidate();

    if (!std::isfinite(config.frequencyStep) || config.frequencyStep <= 0.0f) {
        JST_ERROR("[BLOCK_SDR] Frequency step must be finite and positive.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result SdrImpl::configure() {
    moduleConfig->deviceString = deviceString;
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

Result SdrImpl::define() {
    const auto& config = *candidate();

    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "The output buffer containing samples from the radio."));

    Parser::Sequence deviceOptions{Parser::Map{{"label", "None"}, {"value", ""}}};
    bool selectionListed = config.deviceString.empty();
    for (const auto& [label, device] : Radio::Device::Label(Radio::Device::List())) {
        const auto value = device.selector();
        selectionListed = selectionListed || value == config.deviceString;
        deviceOptions.emplace_back(Parser::Map{{"label", label}, {"value", value}});
    }
    if (!selectionListed) {
        deviceOptions.emplace_back(Parser::Map{{"label", "Configured device"},
                                               {"value", config.deviceString}});
    }

    JST_CHECK(defineInterfaceConfig("deviceString",
                                    "Device",
                                    "Select a radio to receive samples. Choose None to disconnect.",
                                    {{"type", "dropdown"}, {"options", std::move(deviceOptions)}}));

    JST_CHECK(defineInterfaceConfig("antenna",
                                    "Antenna",
                                    "Receive antenna port. Changes restart the receiver.",
                                    antennaFormat(config)));

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
                                    sampleRateFormat(config)));

    JST_CHECK(defineInterfaceConfig("automaticGain",
                                    "Automatic Gain",
                                    "Enable automatic gain control on radios that support it.",
                                    {{"type", "bool"}}));

    if (!config.automaticGain) {
        JST_CHECK(defineInterfaceConfig("manualGain",
                                        "Manual Gain",
                                        "Receive gain applied while automatic gain is off.",
                                        manualGainFormat(config)));
    }

    if (Radio::Device::SupportsBiasTee(config.deviceString)) {
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
                                    {{"type", "progressbar"}},
        [this]() -> std::any {
            const F32 fill = moduleImpl ? moduleImpl->getBufferFill() : 0.0f;
            return std::pair<std::string, F32>{jst::fmt::format("{:.1f}%", fill * 100.0f), fill};
        }));

    JST_CHECK(defineInterfaceMetric("bufferLoss",
                                    "Buffer Loss",
                                    "Share of samples overwritten before the chain read them since the stream started. "
                                    "Gaps the radio reported (GAP) show next to it.",
                                    {{"type", "progressbar"}},
        [this]() -> std::any {
            const auto loss = moduleImpl ? moduleImpl->getLoss()
                                         : Modules::SdrImpl::LossSnapshot{};
            auto label = loss.bufferLoss > 0.0 && loss.bufferLoss < 0.0001
                ? std::string("<0.01%") : jst::fmt::format("{:.2f}%", loss.bufferLoss * 100.0);
            if (loss.deviceGaps > 0) {
                label = jst::fmt::format("{} ({} GAP)", label, loss.deviceGaps);
            }
            return std::pair<std::string, F32>{label, static_cast<F32>(loss.bufferLoss)};
        }));

    JST_CHECK(defineInterfaceMetric("throughput",
                                    "Throughput",
                                    "Rate the chain reads next to the rate the radio sends.",
                                    {{"type", "progressbar"}},
        [this]() -> std::any {
            const auto [actual, expected] = moduleImpl ? moduleImpl->getThroughput()
                                                       : std::pair<F32, F32>{0.0f, 0.0f};
            const F32 fraction = expected > 0.0f ? actual / expected : 0.0f;
            return std::pair<std::string, F32>{
                jst::fmt::format("{:.1f} / {:.1f} MB/s", actual, expected), fraction};
        }));

    return Result::SUCCESS;
}

Result SdrImpl::create() {
    const auto result = moduleCreate("sdr", moduleConfig, {});
    JST_CHECK(updateInterfaceConfigFormat("sampleRate", sampleRateFormat(*this)));
    JST_CHECK(result);
    JST_CHECK(moduleExposeOutput("signal", {"sdr", "signal"}));

    moduleImpl = moduleHandle("sdr")->getImpl<Modules::SdrImpl>();
    sampleRate = moduleImpl->sampleRate;
    moduleConfig->sampleRate = sampleRate;
    JST_CHECK(updateInterfaceConfigFormat("antenna", antennaFormat(*this)));

    return reconfigure();
}

Result SdrImpl::reconfigure() {
    if (const auto* sdr = createdModule(*this); sdr && sdr->getGainRange()) {
        manualGain = sdr->manualGain;
        moduleConfig->manualGain = manualGain;
    }

    if (!automaticGain) {
        JST_CHECK(updateInterfaceConfigFormat("manualGain", manualGainFormat(*this)));
    }

    return Result::SUCCESS;
}

const Modules::SdrImpl* SdrImpl::createdModule(const Blocks::Sdr& config) {
    const auto module = moduleHandle("sdr");
    if (!module || module->state() != Module::State::CREATED) {
        return nullptr;
    }
    const auto* sdr = module->getImpl<Modules::SdrImpl>();
    return sdr && sdr->deviceString == config.deviceString ? sdr : nullptr;
}

Parser::Map SdrImpl::manualGainFormat(const Blocks::Sdr& config) {
    const auto* sdr = createdModule(config);
    const auto range = sdr ? sdr->getGainRange() : std::nullopt;
    const F32 minimum = range ? static_cast<F32>(range->minimum) : 0.0f;
    const F32 maximum = range ? static_cast<F32>(range->maximum) : 60.0f;
    return {{"type", "range"}, {"min", minimum}, {"max", maximum}, {"unit", "dB"}};
}

Parser::Map SdrImpl::antennaFormat(const Blocks::Sdr& config) {
    Parser::Sequence options{Parser::Map{{"label", "Default"}, {"value", ""}}};
    bool selectionListed = config.antenna.empty();
    if (const auto* sdr = createdModule(config)) {
        for (const auto& antenna : sdr->listAntennas()) {
            selectionListed = selectionListed || antenna == config.antenna;
            options.emplace_back(Parser::Map{{"label", antenna}, {"value", antenna}});
        }
    }
    if (!selectionListed) {
        options.emplace_back(Parser::Map{{"label", config.antenna}, {"value", config.antenna}});
    }
    return {{"type", "dropdown"}, {"options", std::move(options)}};
}

Parser::Map SdrImpl::sampleRateFormat(const Blocks::Sdr& config) {
    const auto capabilities = Radio::Device::Cached(config.deviceString);
    const auto rates = capabilities ? capabilities->fixedSampleRates() : std::vector<F64>{};
    if (rates.empty()) {
        return {{"type", "float"}, {"unit", "MHz"}, {"scale", 1.0e6f}, {"precision", 3}};
    }
    Parser::Sequence options;
    for (const auto rate : rates) {
        std::string value;
        Parser::TypedToString(static_cast<F32>(rate), value);
        auto label = rate < 1e6 ? jst::fmt::format("{:g} kHz", rate / 1e3)
                                : jst::fmt::format("{:g} MHz", rate / 1e6);
        options.emplace_back(Parser::Map{{"label", std::move(label)}, {"value", std::move(value)}});
    }
    return {{"type", "dropdown"}, {"options", std::move(options)}};
}

JST_REGISTER_BLOCK(SdrImpl, {"sdr"});

}  // namespace Jetstream::Blocks
