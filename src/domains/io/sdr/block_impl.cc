#include <jetstream/domains/io/sdr/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/io/sdr/module.hh>
#include "module_impl.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>

namespace Jetstream::Blocks {

namespace {

std::string FrequencyLabel(const F64 frequency) {
    return frequency < 1e6 ? jst::fmt::format("{:g} kHz", frequency / 1e3)
                           : jst::fmt::format("{:g} MHz", frequency / 1e6);
}

}  // namespace

struct SdrImpl : public Block::Impl, public DynamicConfig<Blocks::Sdr> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;
    Result reconfigure() override;

 protected:
    bool antennasDefined = false;
    bool bandwidthDefined = false;
    std::shared_ptr<Modules::Sdr> moduleConfig = std::make_shared<Modules::Sdr>();
    Modules::SdrImpl* moduleImpl = nullptr;

 private:
    const Modules::SdrImpl* createdModule(const Blocks::Sdr& config);
    Parser::Map channelsFormat(const Blocks::Sdr& config);
    Parser::Map manualGainFormat(const Blocks::Sdr& config);
    Parser::Map antennaFormat(const Blocks::Sdr& config);
    Parser::Map antennasFormat(const Blocks::Sdr& config);
    bool antennasListed(const Blocks::Sdr& config);
    void sizeAntennas();
    Parser::Map bandwidthFormat(const Blocks::Sdr& config);
    Parser::Map sampleFormatFormat(const Blocks::Sdr& config);
    Parser::Map sampleRateFormat(const Blocks::Sdr& config);
};

Result SdrImpl::validate() {
    const auto& config = *candidate();

    if (!std::isfinite(config.frequencyStep) || config.frequencyStep <= 0.0) {
        JST_ERROR("[BLOCK_SDR] Frequency step must be finite and positive.");
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

void SdrImpl::sizeAntennas() {
    if (channels <= 1) {
        return;
    }
    if (antennas.size() > channels) {
        antennas.resize(channels);
    }
    const U64 receivers = std::min(channels, Radio::Device::ReceiveChannels(deviceString));
    if (antennas.size() < receivers) {
        antennas.resize(receivers, antenna);
    }
}

Result SdrImpl::configure() {
    sizeAntennas();

    moduleConfig->deviceString = deviceString;
    moduleConfig->driverArguments = driverArguments;
    moduleConfig->antenna = antenna;
    moduleConfig->antennas = antennas;
    moduleConfig->channels = channels;
    moduleConfig->frequency = frequency;
    moduleConfig->sampleRate = sampleRate;
    moduleConfig->bandwidth = bandwidth;
    moduleConfig->sampleFormat = sampleFormat;
    moduleConfig->automaticGain = automaticGain;
    moduleConfig->manualGain = manualGain;
    moduleConfig->manualGains = manualGains;
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
                                    "The output buffer containing samples from the radio. "
                                    "Channels above 1 add a leading channel axis."));

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

    const auto selection = Radio::Device::Selection::Parse(config.deviceString);
    const bool soapy = selection && selection->driver == "soapy";
    if (soapy) {
        JST_CHECK(defineInterfaceConfig("driverArguments",
                                        "Driver Arguments",
                                        "SoapySDR stream arguments as key=value pairs separated by commas. "
                                        "Changes restart the receiver.",
                                        {{"type", "text"}, {"advanced", true}}));
    }

    if (soapy || config.channels > 1 || Radio::Device::ReceiveChannels(config.deviceString) > 1) {
        JST_CHECK(defineInterfaceConfig("channels",
                                        "Channels",
                                        "Number of synchronized receive channels. More than one adds an "
                                        "outer channel dimension to Output. Changes restart reception.",
                                        channelsFormat(config)));
    }

    if (config.channels == 1) {
        JST_CHECK(defineInterfaceConfig("antenna",
                                        "Antenna",
                                        "Receive antenna port. Changes restart the receiver.",
                                        antennaFormat(config)));
    }

    antennasDefined = antennasListed(config);
    if (antennasDefined) {
        JST_CHECK(defineInterfaceConfig("antennas",
                                        "Antennas",
                                        "Receive antenna port of each receiver in RX order. New entries "
                                        "start at the single-channel Antenna. Changes restart the receiver.",
                                        antennasFormat(config)));
    }

    JST_CHECK(defineInterfaceConfig("frequency",
                                    "Frequency",
                                    "Tuner frequency.",
                                    {{"type", "float"}, {"unit", "MHz"}, {"scale", 1.0e6f},
                                     {"precision", 3}, {"step_config",
                                        std::isfinite(config.frequencyStep) && config.frequencyStep > 0.0
                                            ? std::string("frequencyStep") : std::string{}}}));

    JST_CHECK(defineInterfaceConfig("sampleRate",
                                    "Sample Rate",
                                    "Sampling rate.",
                                    sampleRateFormat(config)));

    bandwidthDefined = Radio::Device::SupportsBandwidth(config.deviceString);
    if (bandwidthDefined) {
        JST_CHECK(defineInterfaceConfig("bandwidth",
                                        "Filter Bandwidth",
                                        "Width of the analog receive filter. Auto lets the radio pick "
                                        "it for the sample rate. Changes restart the receiver.",
                                        bandwidthFormat(config)));
    }

    JST_CHECK(defineInterfaceConfig("sampleFormat",
                                    "Sample Format",
                                    "Type of the output samples. CF32 converts them to floating "
                                    "point. Integer types keep the samples the radio delivers. "
                                    "Changes restart the receiver.",
                                    sampleFormatFormat(config)));

    JST_CHECK(defineInterfaceConfig("automaticGain",
                                    "Automatic Gain",
                                    "Enable automatic gain control on radios that support it.",
                                    {{"type", "bool"}}));

    if (!config.automaticGain && config.channels == 1) {
        JST_CHECK(defineInterfaceConfig("manualGain",
                                        "Manual Gain",
                                        "Receive gain applied while automatic gain is off.",
                                        manualGainFormat(config)));
    }

    if (!config.automaticGain && config.channels > 1) {
        const auto capabilities = Radio::Device::Cached(config.deviceString);
        JST_CHECK(defineInterfaceConfig("manualGains",
                                        "Manual Gains",
                                        "Receive gain of each receiver in RX order, applied while automatic "
                                        "gain is off. Missing entries use the single-channel Manual Gain.",
                                        {{"type", "vector-inline"},
                                         {"value_type", "float"},
                                         {"unit", capabilities ? capabilities->gainUnit : "dB"}}));
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
            return std::pair<std::string, F32>{jst::fmt::format("{:.0f}%", fill * 100.0f), fill};
        }));

    JST_CHECK(defineInterfaceMetric("bufferLoss",
                                    "Buffer Loss",
                                    "Share of samples overwritten before the chain read them since the stream started. "
                                    "Gaps the radio reported (GAP) show next to it.",
                                    {{"type", "progressbar"}, {"warning_above", 0.0f}, {"error_above", 0.01f}},
        [this]() -> std::any {
            const auto loss = moduleImpl ? moduleImpl->getLoss()
                                         : Modules::SdrImpl::LossSnapshot{};
            auto label = loss.bufferLoss > 0.0 && loss.bufferLoss < 0.001
                ? std::string("<0.1%") : jst::fmt::format("{:.1f}%", loss.bufferLoss * 100.0);
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
    JST_CHECK(updateInterfaceConfigFormat("sampleFormat", sampleFormatFormat(*this)));
    if (bandwidthDefined) {
        JST_CHECK(updateInterfaceConfigFormat("bandwidth", bandwidthFormat(*this)));
    }
    JST_CHECK(result);
    JST_CHECK(moduleExposeOutput("signal", {"sdr", "signal"}));

    moduleImpl = moduleHandle("sdr")->getImpl<Modules::SdrImpl>();
    sampleRate = moduleImpl->sampleRate;
    moduleConfig->sampleRate = sampleRate;
    if (channels == 1) {
        JST_CHECK(updateInterfaceConfigFormat("antenna", antennaFormat(*this)));
    }
    const auto selection = Radio::Device::Selection::Parse(deviceString);
    if (selection && selection->driver == "soapy") {
        JST_CHECK(updateInterfaceConfigFormat("channels", channelsFormat(*this)));
    }
    if (antennasDefined) {
        sizeAntennas();
        moduleConfig->antennas = antennas;
        JST_CHECK(updateInterfaceConfigFormat("antennas", antennasFormat(*this)));
    }

    return reconfigure();
}

Result SdrImpl::reconfigure() {
    if (const auto* sdr = createdModule(*this); sdr && sdr->getGainRange()) {
        manualGain = sdr->manualGain;
        moduleConfig->manualGain = manualGain;
    }

    if (!automaticGain && channels == 1) {
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

Parser::Map SdrImpl::channelsFormat(const Blocks::Sdr& config) {
    const U64 available = Radio::Device::ReceiveChannels(config.deviceString);
    Parser::Sequence options;
    const auto option = [&](const U64 count) {
        std::string value;
        Parser::TypedToString(count, value);
        options.emplace_back(Parser::Map{{"label", std::to_string(count)}, {"value", std::move(value)}});
    };
    for (U64 count = 1; count <= available; ++count) {
        option(count);
    }
    if (config.channels > available) {
        option(config.channels);
    }
    return {{"type", "dropdown"}, {"options", std::move(options)}};
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

bool SdrImpl::antennasListed(const Blocks::Sdr& config) {
    if (config.channels <= 1) {
        return false;
    }
    const auto capabilities = Radio::Device::Cached(config.deviceString);
    return !capabilities || !capabilities->antennas.empty();
}

Parser::Map SdrImpl::antennasFormat(const Blocks::Sdr& config) {
    Parser::Sequence options{Parser::Map{{"label", "Default"}, {"value", ""}}};
    std::vector<std::string> listed{""};
    const auto add = [&](const std::string& antenna) {
        if (std::find(listed.begin(), listed.end(), antenna) == listed.end()) {
            listed.push_back(antenna);
            options.emplace_back(Parser::Map{{"label", antenna}, {"value", antenna}});
        }
    };
    if (const auto capabilities = Radio::Device::Cached(config.deviceString)) {
        for (const auto& antenna : capabilities->antennas) {
            add(antenna);
        }
    }
    add(config.antenna);
    for (const auto& antenna : config.antennas) {
        add(antenna);
    }
    return {{"type", "vector"},
            {"value_type", "string"},
            {"item_label", "Antenna"},
            {"options", std::move(options)}};
}

Parser::Map SdrImpl::bandwidthFormat(const Blocks::Sdr& config) {
    static constexpr std::array<F64, 20> StandardWidths = {
        0.2e6, 0.3e6, 0.6e6, 1.0e6, 1.5e6, 1.75e6, 2.0e6, 2.5e6, 3.0e6, 5.0e6,
        6.0e6, 7.0e6, 8.0e6, 10.0e6, 14.0e6, 20.0e6, 28.0e6, 40.0e6, 56.0e6, 80.0e6,
    };
    std::vector<F64> widths;
    if (const auto capabilities = Radio::Device::Cached(config.deviceString)) {
        for (const auto& range : capabilities->bandwidthRanges) {
            if (range.minimum == range.maximum) {
                widths.push_back(range.minimum);
            }
        }
        for (const auto width : StandardWidths) {
            if (Radio::Device::Range::Contains(capabilities->bandwidthRanges, width)) {
                widths.push_back(width);
            }
        }
    }
    if (config.bandwidth > 0.0f) {
        widths.push_back(config.bandwidth);
    }
    std::sort(widths.begin(), widths.end());
    widths.erase(std::unique(widths.begin(), widths.end(), [](const F64 a, const F64 b) {
        return static_cast<F32>(a) == static_cast<F32>(b);
    }), widths.end());

    Parser::Sequence options{Parser::Map{{"label", "Auto"}, {"value", "0"}}};
    for (const auto width : widths) {
        std::string value;
        Parser::TypedToString(static_cast<F32>(width), value);
        options.emplace_back(Parser::Map{{"label", FrequencyLabel(width)},
                                         {"value", std::move(value)}});
    }
    return {{"type", "dropdown"}, {"options", std::move(options)}};
}

Parser::Map SdrImpl::sampleFormatFormat(const Blocks::Sdr& config) {
    std::vector<Radio::Device::Format> formats{{}};
    if (const auto capabilities = Radio::Device::Cached(config.deviceString)) {
        formats = capabilities->formats;
    }
    const auto saved = NameToDataType(config.sampleFormat);
    if (std::ranges::none_of(formats, [&](const auto& format) { return format.type == saved; })) {
        formats.push_back({saved, 0.0});
    }

    Parser::Sequence options;
    for (const auto& format : formats) {
        std::string value(format.type == DataType::None ? config.sampleFormat
                                                        : DataTypeToName(format.type));
        auto label = value;
        if (format.type != DataType::CF32 && format.type != DataType::CF64 &&
            format.fullScale >= 2.0) {
            label = jst::fmt::format("{} ({:g}-bit)", value, std::log2(format.fullScale) + 1.0);
        }
        options.emplace_back(Parser::Map{{"label", std::move(label)}, {"value", std::move(value)}});
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
        options.emplace_back(Parser::Map{{"label", FrequencyLabel(rate)},
                                         {"value", std::move(value)}});
    }
    return {{"type", "dropdown"}, {"options", std::move(options)}};
}

JST_REGISTER_BLOCK(SdrImpl, {"sdr"});

}  // namespace Jetstream::Blocks
