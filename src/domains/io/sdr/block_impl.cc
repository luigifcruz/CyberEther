#include <jetstream/domains/io/sdr/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/io/sdr/module.hh>
#include "module_impl.hh"

#include <cmath>
#include <limits>

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
    moduleConfig->channels = channels;
    moduleConfig->automaticGains = automaticGains;
    moduleConfig->manualGains = manualGains;
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
    const bool gainOverrides = config.channels > 1 ||
        !config.automaticGains.empty() || !config.manualGains.empty();

    JST_CHECK(
        defineInterfaceOutput("signal", "Output",
                              "Samples with channel, batch, and sample axes; "
                              "single-channel output has batch and sample axes."));

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

    JST_CHECK(defineInterfaceConfig(
        "channels", "Channels",
        "Number of synchronized receive channels. More than one adds an outer "
        "channel dimension to Output. Changes restart reception.",
        {{"type", "uint"}, {"unit", "channels"}}));
    if (config.channels == 1) {
        JST_CHECK(defineInterfaceConfig(
            "antenna", "Antenna", "Receive antenna port. Changes restart the receiver.",
            antennaFormat(config)));
    }

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

    JST_CHECK(defineInterfaceConfig(
        "automaticGain",
        gainOverrides ? "Default Automatic Gain" : "Automatic Gain",
        "Automatic gain used by channels without an Automatic Gains override.",
        {{"type", "bool"}}));

    if (!config.automaticGain || gainOverrides) {
        JST_CHECK(defineInterfaceConfig(
            "manualGain", gainOverrides ? "Default Manual Gain" : "Manual Gain",
            "Manual gain used by channels without a Manual Gains override.",
            manualGainFormat(config)));
    }
    if (gainOverrides) {
        JST_CHECK(defineInterfaceConfig(
            "automaticGains", "Automatic Gains",
            "Values in RX order: 1 for automatic gain, 0 for manual. Missing entries "
            "use Default Automatic Gain. Lists resize when Channels changes.",
            {{"type", "vector-inline"}, {"value_type", "uint"}}));
        const auto caps = Radio::Device::Cached(config.deviceString);
        JST_CHECK(defineInterfaceConfig(
            "manualGains", "Manual Gains",
            "Gains in RX order, clamped to each receiver range. Missing entries use "
            "Default Manual Gain. Lists resize when Channels changes.",
            {{"type", "vector-inline"},
             {"value_type", "float"},
             {"unit", caps ? caps->gainUnit : "dB"}}));
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
                                    {{"type", "progressbar"}, {"warning_above", 0.5f}, {"error_above", 0.75f}},
        [this]() -> std::any {
            const F32 fill = moduleImpl ? moduleImpl->getBufferFill() : 0.0f;
            return std::pair<std::string, F32>{jst::fmt::format("{:.1f}%", fill * 100.0f), fill};
        }));

    JST_CHECK(defineInterfaceMetric("bufferLoss",
                                    "Buffer Loss",
                                    "Share of samples overwritten before the chain read them since the stream started. "
                                    "Gaps the radio reported (GAP) show next to it.",
                                    {{"type", "progressbar"}, {"warning_above", 0.0f}, {"error_above", 0.01f}},
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

Result SdrImpl::create() {
    const auto result = moduleCreate("sdr", moduleConfig, {});
    JST_CHECK(updateInterfaceConfigFormat("sampleRate", sampleRateFormat(*this)));
    JST_CHECK(result);
    JST_CHECK(moduleExposeOutput("signal", {"sdr", "signal"}));

    moduleImpl = moduleHandle("sdr")->getImpl<Modules::SdrImpl>();
    sampleRate = moduleImpl->sampleRate;
    moduleConfig->sampleRate = sampleRate;
    if (channels == 1) {
        JST_CHECK(updateInterfaceConfigFormat("antenna", antennaFormat(*this)));
    }

    return reconfigure();
}

Result SdrImpl::reconfigure() {
    if (const auto* sdr = createdModule(*this); sdr && sdr->getGainRange()) {
        manualGain = sdr->manualGain;
        moduleConfig->manualGain = manualGain;
    }

    if (const auto* sdr = createdModule(*this)) {
        automaticGains = sdr->automaticGains;
        manualGains = sdr->manualGains;
        moduleConfig->automaticGains = automaticGains;
        moduleConfig->manualGains = manualGains;
    }
    if (!automaticGain || channels > 1 ||
        !automaticGains.empty() || !manualGains.empty()) {
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
    const auto capabilities = Radio::Device::Cached(config.deviceString);
    const auto unit = capabilities ? capabilities->gainUnit : "dB";
    return {{"type", "range"}, {"min", minimum}, {"max", maximum}, {"unit", unit}};
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
