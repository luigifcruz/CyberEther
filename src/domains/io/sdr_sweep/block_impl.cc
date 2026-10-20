#include <jetstream/domains/io/sdr_sweep/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/io/sdr_sweep/module.hh>
#include "module_impl.hh"

namespace Jetstream::Blocks {

struct SdrSweepImpl : public Block::Impl, public DynamicConfig<Blocks::SdrSweep> {
    Result configure() override;
    Result define() override;
    Result create() override;
    Result reconfigure() override;

 protected:
    std::shared_ptr<Modules::SdrSweep> moduleConfig = std::make_shared<Modules::SdrSweep>();
    Modules::SdrSweepImpl* moduleImpl = nullptr;

 private:
    const Modules::SdrSweepImpl* createdModule(const Blocks::SdrSweep& config);
    Parser::Map manualGainFormat(const Blocks::SdrSweep& config);
    Parser::Map antennaFormat(const Blocks::SdrSweep& config);
    Parser::Map sampleRateFormat(const Blocks::SdrSweep& config);
};

Result SdrSweepImpl::configure() {
    moduleConfig->deviceString = deviceString;
    moduleConfig->antenna = antenna;
    moduleConfig->automaticGain = automaticGain;
    moduleConfig->manualGain = manualGain;
    moduleConfig->biasTee = biasTee;
    moduleConfig->startFrequency = startFrequency;
    moduleConfig->endFrequency = endFrequency;
    moduleConfig->sampleRate = sampleRate;
    moduleConfig->resolution = resolution;
    moduleConfig->dwell = dwell;
    moduleConfig->settlingTime = settlingTime;

    return Result::SUCCESS;
}

Result SdrSweepImpl::define() {
    const auto& config = *candidate();

    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "Synthetic complex samples whose spectrum is the stitched "
                                    "sweep."));

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
                                    "Select a radio to sweep. Choose None to disconnect.",
                                    {{"type", "dropdown"}, {"options", std::move(deviceOptions)}}));

    JST_CHECK(defineInterfaceConfig("antenna",
                                    "Antenna",
                                    "Receive antenna port.",
                                    antennaFormat(config)));

    JST_CHECK(defineInterfaceConfig("startFrequency",
                                    "Start Frequency",
                                    "Lower edge of the sweep.",
                                    {{"type", "float"}, {"unit", "MHz"}, {"scale", 1.0e6f},
                                     {"precision", 3}}));

    JST_CHECK(defineInterfaceConfig("endFrequency",
                                    "End Frequency",
                                    "Upper edge of the sweep.",
                                    {{"type", "float"}, {"unit", "MHz"}, {"scale", 1.0e6f},
                                     {"precision", 3}}));

    JST_CHECK(defineInterfaceConfig("resolution",
                                    "Resolution",
                                    "Target bin width. Automatically coarsened when capture or "
                                    "output limits require it.",
                                    {{"type", "float"}, {"unit", "kHz"}, {"scale", 1.0e3f},
                                     {"precision", 3}}));

    JST_CHECK(defineInterfaceConfig("sampleRate",
                                    "Capture Rate",
                                    "Hardware sample rate of each hop. Zero uses the radio's "
                                    "preferred rate.",
                                    sampleRateFormat(config)));

    JST_CHECK(defineInterfaceConfig("dwell",
                                    "Dwell",
                                    "Captures per hop. Their power is averaged.",
                                    {{"type", "uint"}, {"unit", "captures"}}));

    JST_CHECK(defineInterfaceConfig("settlingTime",
                                    "Settling Time",
                                    "Time skipped after each retune, once samples queued before "
                                    "it are dropped. Zero uses the radio's default. A nonzero "
                                    "override makes a HackRF retune instead of running a firmware "
                                    "sweep.",
                                    {{"type", "float"}, {"unit", "ms"}, {"precision", 3}}));

    JST_CHECK(defineInterfaceConfig("automaticGain",
                                    "Automatic Gain",
                                    "Lets the radio set its gain. Hop levels are then not "
                                    "comparable.",
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

    JST_CHECK(defineInterfaceMetric("bufferHealth",
                                    "Buffer Health",
                                    "Smoothed occupancy of the single latest-completed-frame "
                                    "software slot, not the radio's hardware FIFO.",
                                    {{"type", "progressbar"}},
        [this]() -> std::any {
            const F32 health = moduleImpl ? moduleImpl->getMetrics().bufferHealth : 0.0f;
            return std::pair<std::string, F32>{jst::fmt::format("{:.0f}%", health * 100.0f), health};
        }));

    JST_CHECK(defineInterfaceMetric("bufferLoss",
                                    "Buffer Loss",
                                    "Completed output frames overwritten or expired before "
                                    "consumption, as a share of completed frames produced. "
                                    "Settling skips and failed hops are not software-buffer loss.",
                                    {{"type", "progressbar"}},
        [this]() -> std::any {
            const F64 loss = moduleImpl ? moduleImpl->getMetrics().bufferLoss : 0.0;
            const auto label = loss > 0.0 && loss < 0.001
                ? std::string("<0.1%") : jst::fmt::format("{:.1f}%", loss * 100.0);
            return std::pair<std::string, F32>{label, static_cast<F32>(loss)};
        }));

    JST_CHECK(defineInterfaceMetric("sweepSpeed",
                                    "Sweep Speed",
                                    "Spectrum covered per second in GHz/s, counting overlapping "
                                    "frequencies only once. The value in parentheses is the number "
                                    "of complete sweeps per second. Both are smoothed and include "
                                    "capture and processing time.",
                                    {{"type", "label"}},
        [this]() -> std::any {
            const auto metrics = moduleImpl ? moduleImpl->getMetrics()
                                             : Modules::SdrSweepImpl::MetricSnapshot{};
            return jst::fmt::format("{:.2f} GHz/s ({:.1f} Hz)", metrics.sweepSpeed,
                                    metrics.sweepRate);
        }));

    return Result::SUCCESS;
}

Result SdrSweepImpl::create() {
    const auto result = moduleCreate("sdr_sweep", moduleConfig, {});
    JST_CHECK(updateInterfaceConfigFormat("sampleRate", sampleRateFormat(*this)));
    JST_CHECK(result);
    JST_CHECK(moduleExposeOutput("signal", {"sdr_sweep", "signal"}));

    moduleImpl = moduleHandle("sdr_sweep")->getImpl<Modules::SdrSweepImpl>();
    JST_CHECK(updateInterfaceConfigFormat("antenna", antennaFormat(*this)));

    return reconfigure();
}

Result SdrSweepImpl::reconfigure() {
    if (const auto* sweep = createdModule(*this); sweep && sweep->getGainRange()) {
        manualGain = sweep->manualGain;
        moduleConfig->manualGain = manualGain;
    }

    if (!automaticGain) {
        JST_CHECK(updateInterfaceConfigFormat("manualGain", manualGainFormat(*this)));
    }

    return Result::SUCCESS;
}

const Modules::SdrSweepImpl* SdrSweepImpl::createdModule(const Blocks::SdrSweep& config) {
    const auto module = moduleHandle("sdr_sweep");
    if (!module || module->state() != Module::State::CREATED) {
        return nullptr;
    }
    const auto* sweep = module->getImpl<Modules::SdrSweepImpl>();
    return sweep && sweep->deviceString == config.deviceString ? sweep : nullptr;
}

Parser::Map SdrSweepImpl::manualGainFormat(const Blocks::SdrSweep& config) {
    const auto* sweep = createdModule(config);
    const auto range = sweep ? sweep->getGainRange() : std::nullopt;
    const F32 minimum = range ? static_cast<F32>(range->minimum) : 0.0f;
    const F32 maximum = range ? static_cast<F32>(range->maximum) : 60.0f;
    return {{"type", "range"}, {"min", minimum}, {"max", maximum}, {"unit", "dB"}};
}

Parser::Map SdrSweepImpl::antennaFormat(const Blocks::SdrSweep& config) {
    Parser::Sequence options{Parser::Map{{"label", "Default"}, {"value", ""}}};
    bool selectionListed = config.antenna.empty();
    if (const auto* sweep = createdModule(config)) {
        for (const auto& antenna : sweep->listAntennas()) {
            selectionListed = selectionListed || antenna == config.antenna;
            options.emplace_back(Parser::Map{{"label", antenna}, {"value", antenna}});
        }
    }
    if (!selectionListed) {
        options.emplace_back(Parser::Map{{"label", config.antenna}, {"value", config.antenna}});
    }
    return {{"type", "dropdown"}, {"options", std::move(options)}};
}

Parser::Map SdrSweepImpl::sampleRateFormat(const Blocks::SdrSweep& config) {
    const auto capabilities = Radio::Device::Cached(config.deviceString);
    const auto rates = capabilities ? capabilities->fixedSampleRates() : std::vector<F64>{};
    if (rates.empty()) {
        return {{"type", "float"}, {"unit", "MHz"}, {"scale", 1.0e6f}, {"precision", 3}};
    }
    std::string preferred;
    Parser::TypedToString(0.0, preferred);
    Parser::Sequence options{Parser::Map{{"label", "Preferred"}, {"value", std::move(preferred)}}};
    for (const auto rate : rates) {
        std::string value;
        Parser::TypedToString(rate, value);
        auto label = rate < 1e6 ? jst::fmt::format("{:g} kHz", rate / 1e3)
                                : jst::fmt::format("{:g} MHz", rate / 1e6);
        options.emplace_back(Parser::Map{{"label", std::move(label)}, {"value", std::move(value)}});
    }
    return {{"type", "dropdown"}, {"options", std::move(options)}};
}

JST_REGISTER_BLOCK(SdrSweepImpl, {"sdr_sweep"});

}  // namespace Jetstream::Blocks
