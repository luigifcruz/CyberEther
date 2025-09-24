#ifndef JETSTREAM_MODULES_PSK_DEMOD_HH
#define JETSTREAM_MODULES_PSK_DEMOD_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory2/tensor.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_PSK_DEMOD_CPU(MACRO) \
    MACRO(PskDemod, CPU, CF32)

JST_SERDES_ENUM(PskType, BPSK, QPSK, PSK8);

template<Device D, typename T = CF32>
class PskDemod : public Module, public Compute {
 public:
    PskDemod();
    ~PskDemod();

    // Configuration

    struct Config {
        PskType pskType = PskType::QPSK;
        F64 sampleRate = 1000000.0;
        F64 symbolRate = 125000.0;
        F64 frequencyLoopBandwidth = 0.05;
        F64 timingLoopBandwidth = 0.05;
        F64 dampingFactor = 0.707;
        F64 excessBandwidth = 0.35;
        U64 bufferSize = 8192;

        JST_SERDES(pskType, sampleRate, symbolRate, frequencyLoopBandwidth,
                   timingLoopBandwidth, dampingFactor, excessBandwidth, bufferSize);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        mem2::Tensor buffer;

        JST_SERDES_INPUT(buffer);
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
        return Taint::DISCONTIGUOUS;
    }

    void info() const final;

    // Constructor

    Result create();

    // Runtime Configuration Functions

    Result setFrequencyLoopBandwidth(F64& frequencyLoopBandwidth);
    Result setTimingLoopBandwidth(F64& timingLoopBandwidth);
    Result setDampingFactor(F64& dampingFactor);

 protected:
    Result createCompute(const Context& ctx) final;
    Result compute(const Context& ctx) final;

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_PSK_DEMOD_CPU_AVAILABLE
JST_PSK_DEMOD_CPU(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

template <> struct jst::fmt::formatter<Jetstream::PskType> : ostream_formatter {};

#endif
