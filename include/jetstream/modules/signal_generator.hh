#ifndef JETSTREAM_MODULES_SIGNAL_GENERATOR_HH
#define JETSTREAM_MODULES_SIGNAL_GENERATOR_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory2/tensor.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_SIGNAL_GENERATOR_CPU(MACRO) \
    MACRO(SignalGenerator, CPU, CF32) \
    MACRO(SignalGenerator, CPU, F32)

JST_SERDES_ENUM(SignalType, Sine, Cosine, Square, Sawtooth, Triangle, Noise, DC, Chirp);

template<Device D, typename T = CF32>
class SignalGenerator : public Module, public Compute {
 public:
    SignalGenerator();
    ~SignalGenerator();

    // Configuration

    struct Config {
        SignalType signalType = SignalType::Sine;
        F64 sampleRate = 1000000.0;
        F64 frequency = 1000.0;
        F64 amplitude = 1.0;
        F64 phase = 0.0;
        F64 dcOffset = 0.0;
        F64 noiseVariance = 1.0;
        F64 chirpStartFreq = 1000.0;
        F64 chirpEndFreq = 10000.0;
        F64 chirpDuration = 1.0;
        U64 bufferSize = 8192;

        JST_SERDES(signalType, sampleRate, frequency, amplitude, phase, dcOffset,
                   noiseVariance, chirpStartFreq, chirpEndFreq, chirpDuration, bufferSize);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        JST_SERDES_INPUT();
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        mem2::Tensor buffer;

        JST_SERDES_OUTPUT(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const mem2::Tensor& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    constexpr Taint taint() const {
        return Taint::CLEAN;
    }

    void info() const final;

    // Constructor

    Result create();

    // Runtime Configuration Functions

    Result setFrequency(F64& frequency);
    Result setSampleRate(F64& sampleRate);
    Result setAmplitude(F64& amplitude);
    Result setPhase(F64& phase);
    Result setDcOffset(F64& dcOffset);
    Result setNoiseVariance(F64& noiseVariance);
    Result setChirpStartFreq(F64& chirpStartFreq);
    Result setChirpEndFreq(F64& chirpEndFreq);
    Result setChirpDuration(F64& chirpDuration);

 protected:
    Result createCompute(const Context& ctx) final;
    Result compute(const Context& ctx) final;

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_SIGNAL_GENERATOR_CPU_AVAILABLE
JST_SIGNAL_GENERATOR_CPU(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

template <> struct jst::fmt::formatter<Jetstream::SignalType> : ostream_formatter {};

#endif
