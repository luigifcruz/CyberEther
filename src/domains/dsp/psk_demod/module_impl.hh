#ifndef JETSTREAM_DOMAINS_DSP_PSK_DEMOD_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_PSK_DEMOD_MODULE_IMPL_HH

#include <deque>

#include <jetstream/domains/dsp/psk_demod/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct PskDemodImpl : public Module::Impl, public DynamicConfig<PskDemod> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result reconfigure() override;

 protected:
    Tensor input;
    Tensor output;

    // Configuration-derived values.
    U64 samplesPerSymbol = 0;
    U64 constellationOrder = 0;
    U64 outputSize = 0;

    // PLL state for frequency/phase recovery.
    F64 phaseAccumulator = 0.0;
    F64 frequencyError = 0.0;
    F64 freqAlpha = 0.0;
    F64 freqBeta = 0.0;

    // Timing recovery state.
    F64 timingAlpha = 0.0;
    F64 timingBeta = 0.0;
    F64 timingMu = 0.0;
    F64 timingOmega = 0.0;
    F64 timingOmegaNominal = 0.0;
    F64 timingOmegaMin = 0.0;
    F64 timingOmegaMax = 0.0;
    U64 timingIndex = 0;

    // Symbol history for MM detector.
    bool hasLastSymbol = false;
    CF32 lastSymbol = CF32{0.0f, 0.0f};
    CF32 lastDecision = CF32{0.0f, 0.0f};

    // Raw sample history for interpolation across buffers.
    std::deque<CF32> sampleHistory;

    // Safety parameters.
    static constexpr F64 MAX_TIMING_ERROR = 1.0;
    static constexpr F64 MIN_TIMING_ERROR = -1.0;
    static constexpr F64 MAX_FREQUENCY_ERROR = 1.0;
    static constexpr F64 MIN_FREQUENCY_ERROR = -1.0;

    // Helper methods.
    void updateLoopCoefficients();
    void initializeState();
    CF32 interpolate(const CF32& a, const CF32& b, F64 mu) const;
    CF32 decision(const CF32& sample) const;
    F64 muellerMullerError(const CF32& prevSymbol, const CF32& prevDecision,
                           const CF32& currentSymbol, const CF32& currentDecision) const;
    F64 costasLoopError(const CF32& sample) const;
    CF32 correctFrequency(const CF32& sample, F64 phase) const;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_PSK_DEMOD_MODULE_IMPL_HH
