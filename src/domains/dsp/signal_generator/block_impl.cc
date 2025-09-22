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

    if (signalType != config.signalType) {
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
    JST_CHECK(defineInterfaceOutput("signal",
                                    "Output",
                                    "The generated signal."));

    JST_CHECK(defineInterfaceConfig("signalType",
                                    "Signal Type",
                                    "The type of signal to generate.",
                                    "dropdown:sine(Sine),cosine(Cosine),square(Square),triangle(Triangle),sawtooth(Sawtooth),noise(Noise),dc(DC),chirp(Chirp)"));

    JST_CHECK(defineInterfaceConfig("signalDataType",
                                    "Data Type",
                                    "The data type for the generated signal samples.",
                                    "dropdown:F32(F32),CF32(CF32)"));

    JST_CHECK(defineInterfaceConfig("bufferSize",
                                    "Buffer Size",
                                    "Number of samples to generate per processing cycle.",
                                    "int:samples"));

    JST_CHECK(defineInterfaceConfig("sampleRate",
                                    "Sample Rate",
                                    "The sampling frequency in MHz.",
                                    "float:MHz:2"));

    JST_CHECK(defineInterfaceConfig("frequency",
                                    "Frequency",
                                    "The fundamental frequency of the generated signal in MHz.",
                                    "float:MHz:2"));

    JST_CHECK(defineInterfaceConfig("amplitude",
                                    "Amplitude",
                                    "The amplitude scaling factor for the signal.",
                                    "float:dBFS:2"));

    JST_CHECK(defineInterfaceConfig("phase",
                                    "Phase",
                                    "Phase offset in radians.",
                                    "float:rad:2"));

    JST_CHECK(defineInterfaceConfig("dcOffset",
                                    "DC Offset",
                                    "DC bias added to the signal.",
                                    "float::2"));

    JST_CHECK(defineInterfaceConfig("noiseVariance",
                                    "Noise Variance",
                                    "Variance of Gaussian noise (for noise signal type).",
                                    "float::2"));

    if (signalType == "chirp") {
        JST_CHECK(defineInterfaceConfig("chirpStartFreq",
                                        "Chirp Start Frequency",
                                        "Start frequency for chirp signals in MHz.",
                                        "float:MHz:2"));

        JST_CHECK(defineInterfaceConfig("chirpEndFreq",
                                        "Chirp End Frequency",
                                        "End frequency for chirp signals in MHz.",
                                        "float:MHz:2"));

        JST_CHECK(defineInterfaceConfig("chirpDuration",
                                        "Chirp Duration",
                                        "Duration of one chirp sweep in seconds.",
                                        "float:sec:2"));
    }

    return Result::SUCCESS;
}

Result SignalGeneratorImpl::create() {
    JST_CHECK(moduleCreate("signal_generator", signalGeneratorConfig, {}));
    JST_CHECK(moduleExposeOutput("signal", {"signal_generator", "signal"}));

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(SignalGeneratorImpl);

}  // namespace Jetstream::Blocks
