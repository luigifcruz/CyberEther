#ifndef JETSTREAM_DOMAINS_DSP_FM_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_FM_MODULE_IMPL_HH

#include <jetstream/domains/dsp/fm/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/memory/axis.hh>

#include <array>
#include <vector>

namespace Jetstream::Modules {

struct FmImpl : public Module::Impl, public DynamicConfig<FM> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    struct Biquad {
        F32 b0 = 0.0f;
        F32 b1 = 0.0f;
        F32 b2 = 0.0f;
        F32 a1 = 0.0f;
        F32 a2 = 0.0f;
    };

    struct BiquadState {
        F32 z1 = 0.0f;
        F32 z2 = 0.0f;
    };

    struct StereoState {
        F32 pilotPhase = 0.0f;
        F32 pilotCosStage = 0.0f;
        F32 pilotSinStage = 0.0f;
        F32 pilotCos = 0.0f;
        F32 pilotSin = 0.0f;
        F32 leftDeemphasis = 0.0f;
        F32 rightDeemphasis = 0.0f;
        BiquadState sumNotch;
        BiquadState differenceNotch;
        std::array<BiquadState, 3> sumFilter;
        std::array<BiquadState, 3> differenceFilter;
    };

    Tensor input;
    Tensor output;
    bool wideBand = false;
    bool deemphasisEnabled = false;
    F32 kf = 0.0f;
    F32 ref = 0.0f;
    F32 pilotPhaseIncrement = 0.0f;
    F32 pilotAlpha = 0.0f;
    F32 deemphasisAlpha = 0.0f;
    Biquad pilotNotch;
    std::array<Biquad, 3> audioLowPass;

    SignalAxes validatedSignalAxes;
    U64 validatedLaneCount = 0;

    SignalAxes signalAxes;
    U64 laneCount = 0;
    std::vector<CF32> previousSample;
    std::vector<U8> hasPreviousSample;
    std::vector<F32> narrowDeemphasisState;
    std::vector<StereoState> stereoState;

    void updateCoefficients();
    F32 applyBiquad(F32 sample, const Biquad& coefficients,
                    BiquadState& state) const;
    F32 applyAudioLowPass(F32 sample,
                          std::array<BiquadState, 3>& state) const;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_FM_MODULE_IMPL_HH
