#include <jetstream/domains/io/soapy/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/io/soapy/module.hh>
#include "module_impl.hh"
#include "library/soapysdr.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace Jetstream::Blocks {

namespace {

template<typename T>
Parser::Map DeviceStatus(Parser::Map format, const Modules::SoapyQuery<T>& query,
                         const T& requested, const char* unit = "", F64 scale = 1.0) {
    if (!query.error.empty()) {
        format["hint"] = "Device: " + query.error;
        format["hint_tone"] = std::string("error");
    } else if (query.value) {
        const auto& value = *query.value;
        bool changed = value != requested;
        std::string text;
        if constexpr (std::is_floating_point_v<T>) {
            changed = std::abs(value - requested) >
                std::max(1.0e-6, std::abs(requested) * std::numeric_limits<F32>::epsilon());
            text = jst::fmt::format("{:.9g} {}", value / scale, unit);
        } else if constexpr (std::is_same_v<T, bool>) {
            text = value ? "On" : "Off";
        } else {
            text = value;
        }
        format["hint"] = "Device: " + text;
        format["hint_tone"] = std::string(changed ? "warning" : "secondary");
    }
    return format;
}

}  // namespace

struct SoapyImpl : public Block::Impl, public DynamicConfig<Blocks::Soapy> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;
    Result reconfigure() override;

 protected:
    std::shared_ptr<Modules::Soapy> moduleConfig = std::make_shared<Modules::Soapy>();
    Modules::SoapyImpl* moduleImpl = nullptr;
    std::vector<std::string> deviceControls;

 private:
    Result updateDeviceInterface();
    Result defineControl(const std::string& key, const std::string& label,
                         const std::string& help, const Parser::Map& format);
    Result defineDeviceControls(const Modules::SoapyDeviceInfo& info,
                                const Blocks::Soapy& config);
    Result defineArguments(const std::string& map,
        const Modules::SoapyQuery<std::vector<Modules::SoapyCapabilities::Argument>>& args,
        const Modules::SoapyDeviceInfo& info);
    Modules::SoapyDeviceInfo deviceInfo();
    Parser::Map dataTypeFormat(const std::string& selected,
                              const Modules::SoapyDeviceInfo& info);
    Parser::Map antennaFormat(const std::string& selected,
                              const Modules::SoapyDeviceInfo& info);
    Parser::Map manualGainFormat(const Modules::SoapyDeviceInfo& info);

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
    moduleConfig->dataType = dataType;
    moduleConfig->antenna = antenna;
    moduleConfig->frequency = frequency;
    moduleConfig->sampleRate = sampleRate;
    moduleConfig->automaticGain = automaticGain;
    moduleConfig->manualGain = manualGain;
    moduleConfig->gainMode = gainMode;
    moduleConfig->gains = gains;
    moduleConfig->bandwidth = bandwidth;
    moduleConfig->settings = settings;
    moduleConfig->streamArgs = streamArgs;
    moduleConfig->biasTee = biasTee;
    moduleConfig->numberOfBatches = numberOfBatches;
    moduleConfig->numberOfTimeSamples = numberOfTimeSamples;
    moduleConfig->bufferMultiplier = bufferMultiplier;

    return Result::SUCCESS;
}

Result SoapyImpl::define() {
    const auto& config = *candidate();
    const auto info = deviceInfo();

    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "The output buffer containing samples from the SDR device."));

    JST_CHECK(defineInterfaceMetric("actualSettings",
                                    "Device Status",
                                    "Values read after the last control change. "
                                    "Automatic gain may change between readbacks.",
                                    {{"type", "table"}, {"visibility", "internal"}},
        [this]() -> std::any {
            const auto info = moduleImpl ? moduleImpl->getDeviceInfo()
                                         : Modules::SoapyDeviceInfo{};
            std::vector<std::vector<std::string>> rows;
            if (info.connected) {
                const auto number = [](const auto& query, const char* unit) {
                    return query.value
                        ? jst::fmt::format("{:.9g} {}", *query.value, unit)
                        : (query.error.empty() ? "Unavailable" : query.error);
                };
                const auto& state = info.state;
                rows = {{"Frequency", number(state.frequency, "Hz")},
                        {"Sample Rate", number(state.sampleRate, "S/s")},
                        {"Bandwidth", number(state.bandwidth, "Hz")},
                        {"Gain", number(state.gain, "dB")},
                        {"Antenna", state.antenna.value.value_or("Unavailable")},
                        {"Automatic Gain", state.automaticGain.value
                            ? (*state.automaticGain.value ? "On" : "Off")
                            : "Unavailable"}};
                for (const auto& [name, gain] : state.gains) {
                    rows.push_back({name + " Gain", number(gain, "dB")});
                }
                for (const auto& [name, setting] : state.settings) {
                    rows.push_back({"Setting: " + name,
                        setting.value.value_or(setting.error.empty()
                            ? "Unavailable" : setting.error)});
                }
                const auto& caps = info.capabilities;
                for (const auto& [label, error] : std::vector<std::pair<std::string, std::string>>{
                         {"Frequency Range", caps.frequencies.error},
                         {"Sample Rate Range", caps.sampleRates.error},
                         {"Gain Range", caps.gain.error},
                         {"Gain Stages", caps.gains.error},
                         {"Bandwidth Range", caps.bandwidths.error},
                         {"Driver Settings", caps.settings.error},
                         {"Stream Arguments", caps.streamArgs.error},
                         {"Gain Mode", caps.automaticGain.error},
                         {"Antennas", caps.antennas.error},
                         {"Bias-T", caps.biasTee.error}}) {
                    if (!error.empty()) {
                        rows.push_back({label, error});
                    }
                }
                for (const auto& [name, range] : caps.gainRanges) {
                    if (!range.error.empty()) {
                        rows.push_back({name + " Range", range.error});
                    }
                }
            }
            Parser::Map table;
            static_cast<void>(Parser::Serialize(table, "columns",
                                                std::vector<std::string>{"Setting", "Value"}));
            static_cast<void>(Parser::Serialize(table, "rows", rows));
            return table;
        }));

    JST_CHECK(defineInterfaceMetric("bufferFill",
                                    "Buffer Fill",
                                    "How full the internal buffer is. Near 100% means the chain is falling behind.",
                                    {{"type", "progressbar"}, {"warning_above", 0.5f}, {"error_above", 0.75f}},
        [this]() -> std::any {
            if (!moduleImpl) {
                return std::pair<std::string, F32>{"0.0%", 0.0f};
            }
            const F32 bufferFill = moduleImpl->getBufferFill();
            return std::pair<std::string, F32>{jst::fmt::format("{:.1f}%", bufferFill * 100.0f),
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
            auto label = loss > 0.0 && loss < 0.0001
                ? std::string("<0.01%") : jst::fmt::format("{:.2f}%", loss * 100.0);
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

    return defineDeviceControls(info, config);
}

Result SoapyImpl::create() {
    moduleImpl = nullptr;
    JST_CHECK(moduleCreate("soapy", moduleConfig, {}));
    JST_CHECK(moduleExposeOutput("signal", {"soapy", "signal"}));

    moduleImpl = moduleHandle("soapy")->getImpl<Modules::SoapyImpl>();
    return updateDeviceInterface();
}

Result SoapyImpl::reconfigure() {
    return updateDeviceInterface();
}

Modules::SoapyDeviceInfo SoapyImpl::deviceInfo() {
    const auto module = moduleHandle("soapy");
    const auto& config = *candidate();
    if (!module || module->state() != Module::State::CREATED ||
        config.deviceString != deviceString || config.modulePath != modulePath ||
        config.streamString != streamString || config.antenna != antenna) {
        return {};
    }
    return module->getImpl<Modules::SoapyImpl>()->getDeviceInfo();
}

Result SoapyImpl::updateDeviceInterface() {
    return defineDeviceControls(moduleImpl->getDeviceInfo(), *this);
}

Result SoapyImpl::defineControl(const std::string& key, const std::string& label,
                                const std::string& help, const Parser::Map& format) {
    JST_CHECK(defineInterfaceConfig(key, label, help, format));
    deviceControls.push_back(key);
    return Result::SUCCESS;
}

Result SoapyImpl::defineDeviceControls(const Modules::SoapyDeviceInfo& info,
                                      const Blocks::Soapy& config) {
    for (const auto& key : deviceControls) {
        JST_CHECK(removeInterfaceConfig(key));
    }
    deviceControls.clear();
    const auto& caps = info.capabilities;
    const auto& state = info.state;
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
        deviceOptions.emplace_back(Parser::Map{{"label", "Configured device"},
                                               {"value", config.deviceString}});
    }
    JST_CHECK(defineControl("deviceString", "Device",
        "Select a device to receive samples. Choose None to disconnect.",
        {{"type", "dropdown"}, {"options", std::move(deviceOptions)}}));

    if (!info.connected || (caps.antennas.value && !caps.antennas.value->empty())) {
        JST_CHECK(defineControl("antenna", "Antenna",
            "Receive antenna port. Changes restart the receiver.",
            DeviceStatus(antennaFormat(config.antenna, info), state.antenna,
                         config.antenna.empty() ? state.antenna.value.value_or("")
                                                : config.antenna)));
    }
    JST_CHECK(defineControl("dataType", "Data Type",
        "Sample type delivered by SoapySDR. Changes restart the receiver and must "
        "match downstream inputs.", dataTypeFormat(config.dataType, info)));
    JST_CHECK(defineControl("frequency", "Frequency", "Tuner frequency.",
        DeviceStatus({{"type", "float"}, {"unit", "MHz"}, {"scale", 1.0e6f},
                      {"precision", 3}, {"step_config", "frequencyStep"}},
                     state.frequency, F64{config.frequency}, "MHz", 1.0e6)));
    JST_CHECK(defineControl("sampleRate", "Sample Rate", "Sampling rate.",
        DeviceStatus({{"type", "float"}, {"unit", "MHz"}, {"scale", 1.0e6f},
                      {"precision", 3}}, state.sampleRate,
                     F64{config.sampleRate}, "MHz", 1.0e6)));

    if (!info.connected || caps.automaticGain.value == true) {
        JST_CHECK(defineControl("automaticGain", "Automatic Gain",
            "Enable automatic gain control. Device status is the last readback; "
            "gain may change between control changes.",
            DeviceStatus({{"type", "bool"}}, state.automaticGain, config.automaticGain)));
    }
    const bool manual = caps.automaticGain.value == false || !config.automaticGain;
    if (manual && caps.gains.value && !caps.gains.value->empty()) {
        JST_CHECK(defineControl("gainMode", "Manual Gain Mode",
            "Choose total gain or the gain stages reported by the driver.",
            {{"type", "dropdown"}, {"options", Parser::Sequence{
                Parser::Map{{"label", "Total"}, {"value", "total"}},
                Parser::Map{{"label", "Stages"}, {"value", "stages"}}}}}));
    }
    if (manual && config.gainMode == "total" &&
        (!info.connected || (caps.gain.value &&
         caps.gain.value->minimum != caps.gain.value->maximum))) {
        JST_CHECK(defineControl("manualGain", "Manual Gain",
            "Requested receive gain while automatic gain is off. "
            "The device status below shows the driver's last readback.",
            DeviceStatus(manualGainFormat(info), state.gain, F64{config.manualGain}, "dB")));
    }
    if (manual && config.gainMode == "stages") {
        for (const auto& [name, query] : caps.gainRanges) {
            if (!query.value || query.value->minimum == query.value->maximum) {
                continue;
            }
            const auto& range = *query.value;
            const auto gain = state.gains.find(name);
            const auto readback = gain != state.gains.end()
                ? gain->second : Modules::SoapyQuery<F64>{};
            const F64 value = readback.value.value_or(range.minimum);
            Parser::Map format{{"type", "range"}, {"map", "gains"}, {"key", name},
                               {"default", value}, {"unit", "dB"},
                               {"min", range.minimum}, {"max", range.maximum}};
            if (range.step > 0.0 && std::isfinite(static_cast<F32>(range.step))) {
                format["step"] = range.step;
            }
            JST_CHECK(defineControl("gains:" + name, name + " Gain",
                "Requested stage gain. Device status shows the driver's last readback.",
                DeviceStatus(std::move(format), readback,
                             Parser::Get<F64>(config.gains, name, value), "dB")));
        }
    }
    if (!info.connected || (caps.bandwidths.value && !caps.bandwidths.value->empty())) {
        JST_CHECK(defineControl("bandwidth", "Bandwidth",
            "Hardware receive filter width. Zero keeps the driver's choice. "
            "Returning to zero restarts the receiver.",
            DeviceStatus({{"type", "float"}, {"unit", "MHz"}, {"scale", 1.0e6f},
                          {"precision", 3}, {"min", 0.0f}}, state.bandwidth,
                         config.bandwidth == 0.0 ? state.bandwidth.value.value_or(0.0)
                                                 : config.bandwidth, "MHz", 1.0e6)));
    }
    if (!info.connected || caps.biasTee.value == true) {
        JST_CHECK(defineControl("biasTee", "Bias-T",
            "Enable antenna power through the antenna cable.", {{"type", "bool"}}));
    }
    JST_CHECK(defineControl("numberOfBatches", "Batches",
        "Number of batches in output buffer.", {{"type", "uint"}, {"unit", "batches"}}));
    JST_CHECK(defineControl("numberOfTimeSamples", "Samples",
        "Samples per batch and the FFT size downstream.",
        {{"type", "uint"}, {"unit", "samples"}}));
    JST_CHECK(defineControl("bufferMultiplier", "Buffer Multiplier",
        "Internal buffer size multiplier.", {{"type", "uint"}, {"unit", "x"}}));
    JST_CHECK(defineControl("advanced", "Advanced Parameters",
        "Show device arguments, stream arguments and driver settings. "
        "A custom driver library path can be set through the node's Inspect menu.",
        {{"type", "bool"}}));
    if (config.advanced) {
        JST_CHECK(defineControl("deviceArguments", "Device Arguments",
            "Device creation arguments. Changes reopen the receiver. "
            "Leave empty to disconnect.",
            {{"type", "text"}, {"config", "deviceString"}}));
        JST_CHECK(defineControl("streamString", "Stream Arguments",
            "SoapySDR stream arguments. Typed overrides below take precedence. "
            "Changes restart the receiver.", {{"type", "text"}}));
        JST_CHECK(defineArguments("streamArgs", caps.streamArgs, info));
        JST_CHECK(defineArguments("settings", caps.settings, info));
    }
    return Result::SUCCESS;
}

Result SoapyImpl::defineArguments(const std::string& map,
    const Modules::SoapyQuery<std::vector<Modules::SoapyCapabilities::Argument>>& query,
    const Modules::SoapyDeviceInfo& info) {
    if (!query.value) {
        return Result::SUCCESS;
    }
    for (const auto& arg : *query.value) {
        if (map == "settings" && arg.key == "biastee") {
            continue;
        }
        std::string value = arg.value;
        if (map == "settings") {
            const auto actual = info.state.settings.find(arg.key);
            if (actual != info.state.settings.end() && actual->second.value) {
                value = *actual->second.value;
            }
        }
        if (value.empty() && arg.type != Modules::SoapyAbi::ArgString) {
            value = arg.type == Modules::SoapyAbi::ArgBool ? "false" : "0";
        }
        Parser::Map format{{"map", map}, {"key", arg.key}, {"default", value},
                           {"unit", arg.units}, {"type", "text"}};
        if (!arg.options.empty()) {
            Parser::Sequence options;
            for (U64 i = 0; i < arg.options.size(); ++i) {
                const auto label = i < arg.optionNames.size() && !arg.optionNames[i].empty()
                    ? arg.optionNames[i] : arg.options[i];
                options.emplace_back(Parser::Map{{"label", label}, {"value", arg.options[i]}});
            }
            format["type"] = std::string("dropdown");
            format["options"] = std::move(options);
        } else if (arg.type == Modules::SoapyAbi::ArgBool) {
            format["type"] = std::string("bool");
        } else if (arg.type == Modules::SoapyAbi::ArgInt) {
            format["type"] = std::string("int");
        } else if (arg.type == Modules::SoapyAbi::ArgFloat) {
            format["type"] = std::string("float");
            if (std::isfinite(arg.range.minimum) && std::isfinite(arg.range.maximum) &&
                std::isfinite(static_cast<F32>(arg.range.minimum)) &&
                std::isfinite(static_cast<F32>(arg.range.maximum)) &&
                arg.range.maximum > arg.range.minimum) {
                format["type"] = std::string("range");
                format["min"] = arg.range.minimum;
                format["max"] = arg.range.maximum;
            }
        }
        const bool numeric = arg.type == Modules::SoapyAbi::ArgInt ||
                             arg.type == Modules::SoapyAbi::ArgFloat;
        if (numeric && arg.range.step > 0.0 &&
            std::isfinite(static_cast<F32>(arg.range.step))) {
            format["step"] = arg.range.step;
        }
        const auto key = map + ":" + arg.key;
        const auto label = (map == "streamArgs" ? "Stream: " : "Setting: ") +
                           (arg.name.empty() ? arg.key : arg.name);
        std::string help = arg.description + " Changes restart the receiver.";
        if (numeric && std::isfinite(arg.range.minimum) &&
            std::isfinite(arg.range.maximum) &&
            arg.range.maximum > arg.range.minimum) {
            help += jst::fmt::format(" Range: {} to {} {}.",
                                     arg.range.minimum, arg.range.maximum, arg.units);
        }
        if (map == "settings") {
            const auto state = info.state.settings.find(arg.key);
            if (state != info.state.settings.end()) {
                std::string requested = value;
                const auto& overrides = candidate()->settings;
                if (overrides.contains(arg.key)) {
                    JST_CHECK(Parser::TypedToString(overrides.at(arg.key), requested));
                }
                format = DeviceStatus(std::move(format), state->second, requested);
            }
        }
        JST_CHECK(defineControl(key, label, help, format));
    }
    return Result::SUCCESS;
}

Parser::Map SoapyImpl::dataTypeFormat(const std::string& selected,
                                     const Modules::SoapyDeviceInfo& info) {
    Parser::Sequence options;
    auto types = info.capabilities.dataTypes.value.value_or(std::vector<std::string>{});
    if (!info.connected) {
        for (const auto& format : Modules::SoapySampleFormats) {
            types.emplace_back(DataTypeToName(format.dtype));
        }
    }
    bool selectionListed = false;
    for (const auto& type : types) {
        selectionListed |= type == selected;
        options.emplace_back(Parser::Map{{"label", type}, {"value", type}});
    }
    if (!selectionListed) {
        options.emplace_back(Parser::Map{{"label", selected + " (unsupported)"},
                                         {"value", selected}});
    }
    return {{"type", "dropdown"}, {"options", std::move(options)}};
}

Parser::Map SoapyImpl::antennaFormat(const std::string& selected,
                                    const Modules::SoapyDeviceInfo& info) {
    Parser::Sequence options{Parser::Map{{"label", "Default"}, {"value", ""}}};
    bool selectionListed = selected.empty();
    if (const auto& antennas = info.capabilities.antennas.value) {
        for (const auto& antenna : *antennas) {
            selectionListed |= antenna == selected;
            options.emplace_back(Parser::Map{{"label", antenna}, {"value", antenna}});
        }
    }
    if (!selectionListed) {
        options.emplace_back(Parser::Map{{"label", selected}, {"value", selected}});
    }
    return {{"type", "dropdown"}, {"options", std::move(options)}};
}

Parser::Map SoapyImpl::manualGainFormat(const Modules::SoapyDeviceInfo& info) {
    const auto& range = info.capabilities.gain.value;
    const F32 minimum = range ? static_cast<F32>(range->minimum) : 0.0f;
    const F32 maximum = range ? static_cast<F32>(range->maximum) : 60.0f;
    return {{"type", "range"}, {"min", minimum}, {"max", maximum}, {"unit", "dB"}};
}

JST_REGISTER_BLOCK(SoapyImpl, {"soapy"});

}  // namespace Jetstream::Blocks
