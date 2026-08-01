#ifndef JETSTREAM_DOMAINS_DSP_PSK_DEMOD_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_PSK_DEMOD_MODULE_IMPL_HH

#include <optional>
#include <vector>

#include <jetstream/domains/dsp/psk_demod/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/memory/axis.hh>
#include <jetstream/tools/circular_buffer.hh>

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

    // Candidate configuration derivations populated by validation.
    U64 validatedSamplesPerSymbol = 0;
    U64 validatedConstellationOrder = 0;
    U64 validatedOutputSize = 0;
    U64 validatedOutputSizeBytes = 0;
    U64 validatedMaxIterations = 0;
    U64 validatedInputSampleSize = 0;
    U64 validatedOutputSampleSize = 0;
    U64 validatedSampleHistoryCapacity = 0;
    U64 validatedOutputSymbolsPerLane = 0;
    U64 validatedPendingSymbolsCapacity = 0;
    U64 validatedBatchSize = 1;
    U64 validatedLaneCount = 1;
    SignalAxes validatedSignalAxes;
    Shape validatedOutputShape;
    F64 validatedFreqAlpha = 0.0;
    F64 validatedFreqBeta = 0.0;
    F64 validatedTimingAlpha = 0.0;
    F64 validatedTimingBeta = 0.0;
    F64 validatedTimingOmegaNominal = 0.0;
    F64 validatedTimingOmegaMin = 0.0;
    F64 validatedTimingOmegaMax = 0.0;
    F32 validatedOutputSampleRate = 0.0f;

    // Configuration-derived values.
    U64 samplesPerSymbol = 0;
    U64 constellationOrder = 0;
    U64 outputSize = 0;
    U64 maxIterations = 0;
    U64 inputSampleSize = 0;
    U64 outputSampleSize = 0;
    U64 sampleHistoryCapacity = 0;
    U64 outputSymbolsPerLane = 0;
    U64 pendingSymbolsCapacity = 0;
    U64 batchSize = 1;
    U64 laneCount = 1;
    Index sampleAxis = 0;
    std::optional<Index> batchAxis;
    std::vector<Index> laneAxes;

    // Loop coefficients shared by all independent lanes.
    F64 frequencyError = 0.0;
    F64 freqAlpha = 0.0;
    F64 freqBeta = 0.0;
    F64 timingAlpha = 0.0;
    F64 timingBeta = 0.0;
    F64 timingOmegaNominal = 0.0;
    F64 timingOmegaMin = 0.0;
    F64 timingOmegaMax = 0.0;

    struct DemodState {
        F64 phaseAccumulator = 0.0;
        F64 frequencyError = 0.0;
        F64 timingMu = 0.0;
        F64 timingOmega = 0.0;
        U64 timingIndex = 0;
        bool hasLastSymbol = false;
        CF32 lastSymbol = CF32{0.0f, 0.0f};
        CF32 lastDecision = CF32{0.0f, 0.0f};
        Tools::CircularBuffer<CF32> sampleHistory{0, Tools::CircularBufferOverflowPolicy::Reject};
        Tools::CircularBuffer<CF32> pendingSymbols{0, Tools::CircularBufferOverflowPolicy::Reject};
    };
    std::vector<DemodState> laneStates;

    // Safety parameters.
    static constexpr F64 MAX_TIMING_ERROR = 1.0;
    static constexpr F64 MIN_TIMING_ERROR = -1.0;
    static constexpr F64 MAX_FREQUENCY_ERROR = 1.0;
    static constexpr F64 MIN_FREQUENCY_ERROR = -1.0;

    // Helper methods.
    Result initializeState(DemodState& state);
    CF32 interpolate(const CF32& a, const CF32& b, F64 mu) const;
    CF32 decision(const CF32& sample) const;
    F64 muellerMullerError(const CF32& prevSymbol, const CF32& prevDecision,
                           const CF32& currentSymbol, const CF32& currentDecision) const;
    F64 costasLoopError(const CF32& sample) const;
    CF32 correctFrequency(const CF32& sample, F64 phase) const;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_PSK_DEMOD_MODULE_IMPL_HH
