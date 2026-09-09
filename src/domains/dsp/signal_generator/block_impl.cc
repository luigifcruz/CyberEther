#include <jetstream/domains/dsp/signal_generator/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/dsp/signal_generator/module.hh>

namespace Jetstream::Blocks {

struct SignalGeneratorImpl : public Block::Impl, public DynamicConfig<Blocks::SignalGenerator> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::SignalGenerator> signalGeneratorConfig = std::make_shared<Modules::SignalGenerator>();
};

Result SignalGeneratorImpl::validate() {
    const auto& config = *candidate();

    if (signalType != config.signalType ||
        signalDataType != config.signalDataType ||
        sampleRate != config.sampleRate ||
        bufferSize != config.bufferSize) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

Result SignalGeneratorImpl::configure() {
    signalGeneratorConfig->signalType = signalType;
    signalGeneratorConfig->signalDataType = signalDataType;
    signalGeneratorConfig->bufferSize = bufferSize;
    signalGeneratorConfig->sampleRate = sampleRate;
    signalGeneratorConfig->frequency = frequency;
    signalGeneratorConfig->amplitude = amplitude;
    signalGeneratorConfig->phase = phase;
    signalGeneratorConfig->dcOffset = dcOffset;
    signalGeneratorConfig->noiseVariance = noiseVariance;
    signalGeneratorConfig->chirpStartFreq = chirpStartFreq;
    signalGeneratorConfig->chirpEndFreq = chirpEndFreq;
    signalGeneratorConfig->chirpDuration = chirpDuration;

    return Result::SUCCESS;
}

Result SignalGeneratorImpl::define() {
    const auto& config = *candidate();
    const bool isPeriodic = config.signalType == "sine" ||
                            config.signalType == "cosine" ||
                            config.signalType == "square" ||
                            config.signalType == "triangle" ||
                            config.signalType == "sawtooth";
    const bool isNoise = config.signalType == "noise";
    const bool isDc = config.signalType == "dc";
    const bool isChirp = config.signalType == "chirp";

    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "The generated signal."));

    JST_CHECK(defineInterfaceConfig("signalType",
                                    "Signal Type",
                                    "The type of signal to generate.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "Sine"}, {"value", "sine"}},
                                        Parser::Map{{"label", "Cosine"}, {"value", "cosine"}},
                                        Parser::Map{{"label", "Square"}, {"value", "square"}},
                                        Parser::Map{{"label", "Triangle"}, {"value", "triangle"}},
                                        Parser::Map{{"label", "Sawtooth"}, {"value", "sawtooth"}},
                                        Parser::Map{{"label", "Noise"}, {"value", "noise"}},
                                        Parser::Map{{"label", "DC"}, {"value", "dc"}},
                                        Parser::Map{{"label", "Chirp"}, {"value", "chirp"}},
                                    }}}));

    JST_CHECK(defineInterfaceConfig("signalDataType",
                                    "Data Type",
                                    "CF32 sine, cosine, and chirp use analytic IQ; "
                                    "other waveforms remain real-valued.",
                                    {{"type", "dropdown"}, {"options", Parser::Sequence{
                                        Parser::Map{{"label", "F32"}, {"value", "F32"}},
                                        Parser::Map{{"label", "CF32"}, {"value", "CF32"}},
                                    }}}));

    JST_CHECK(defineInterfaceConfig("sampleRate",
                                    "Sample Rate",
                                    "Sampling frequency. Raw configuration uses Hz.",
                                    {{"type", "float"}, {"unit", "MHz"}, {"scale", 1.0e6f}, {"precision", 3}}));

    if (isPeriodic) {
        const std::string frequencyHelp =
            config.signalDataType == "CF32" &&
            (config.signalType == "sine" || config.signalType == "cosine") ?
            "Signed baseband frequency; limited to +/- Nyquist." :
            "Baseband frequency; limited to the positive Nyquist range.";
        JST_CHECK(defineInterfaceConfig("frequency",
                                        "Frequency",
                                        frequencyHelp,
                                        {{"type", "float"}, {"unit", "MHz"}, {"scale", 1.0e6f}, {"precision", 6}}));
    }

    if (isChirp) {
        const std::string chirpFrequencyHelp =
            config.signalDataType == "CF32" ?
            "Signed baseband frequency; limited to +/- Nyquist." :
            "Baseband frequency; limited to the positive Nyquist range.";
        JST_CHECK(defineInterfaceConfig("chirpStartFreq",
                                        "Start Frequency",
                                        chirpFrequencyHelp,
                                        {{"type", "float"}, {"unit", "MHz"}, {"scale", 1.0e6f}, {"precision", 6}}));

        JST_CHECK(defineInterfaceConfig("chirpEndFreq",
                                        "End Frequency",
                                        chirpFrequencyHelp,
                                        {{"type", "float"}, {"unit", "MHz"}, {"scale", 1.0e6f}, {"precision", 6}}));

        JST_CHECK(defineInterfaceConfig("chirpDuration",
                                        "Duration",
                                        "Duration of each phase-continuous sweep.",
                                        {{"type", "float"}, {"unit", "sec"}, {"precision", 3}}));
    }

    if (isPeriodic || isNoise || isDc || isChirp) {
        const std::string amplitudeLabel = isDc ? "Level" : "Amplitude";
        const std::string amplitudeHelp = isNoise ?
            "Linear noise multiplier; sigma = amplitude * sqrt(variance)." :
            (isDc ? "DC level before the additional DC offset." :
                    "Linear peak amplitude multiplier.");
        JST_CHECK(defineInterfaceConfig("amplitude",
                                        amplitudeLabel,
                                        amplitudeHelp,
                                        {{"type", "float"}, {"precision", 3}}));
    }

    if (isNoise) {
        JST_CHECK(defineInterfaceConfig("noiseVariance",
                                        "Noise Variance",
                                        "Per-component Gaussian variance before "
                                        "amplitude scaling.",
                                        {{"type", "float"}, {"precision", 3}}));
    }

    if (isPeriodic || isChirp) {
        JST_CHECK(defineInterfaceConfig("phase",
                                        "Phase",
                                        "Initial phase offset in radians.",
                                        {{"type", "float"}, {"unit", "rad"}, {"precision", 3}}));
    }

    if (isPeriodic || isNoise || isDc || isChirp) {
        JST_CHECK(defineInterfaceConfig("dcOffset",
                                        isDc ? "Additional Offset" : "DC Offset",
                                        "Real-valued bias added to generated samples.",
                                        {{"type", "float"}, {"precision", 3}}));
    }

    JST_CHECK(defineInterfaceConfig("bufferSize",
                                    "Buffer Size",
                                    "Samples generated per processing cycle.",
                                    {{"type", "uint"}, {"unit", "samples"}}));

    return Result::SUCCESS;
}

Result SignalGeneratorImpl::create() {
    JST_CHECK(moduleCreate("signal_generator", signalGeneratorConfig, {}));
    JST_CHECK(moduleExposeOutput("signal", {"signal_generator", "signal"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(SignalGeneratorImpl, {"signal_generator"});

}  // namespace Jetstream::Blocks
