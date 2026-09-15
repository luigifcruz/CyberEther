#ifndef JETSTREAM_DOMAINS_DSP_RATIONAL_RESAMPLER_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_RATIONAL_RESAMPLER_MODULE_IMPL_HH

#include <jetstream/domains/dsp/rational_resampler/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/memory/axis.hh>

namespace Jetstream::Modules {

struct RationalResamplerImpl : public Module::Impl,
                               public DynamicConfig<RationalResampler> {
    Result validate() override;
    Result define() override;
    Result create() override;
    Result reconfigure() override;

 protected:
    Result updateSampleRate();

    struct Plan {
        SignalAxes axes;
        Shape outputShape;
        U64 interpolation = 1;
        U64 decimation = 1;
        U64 tapCount = 0;
        U64 phaseLength = 0;
        U64 inputSamples = 0;
        U64 outputSamples = 0;
        U64 batchCount = 1;
        U64 laneCount = 0;
        U64 duration = 0;
        U64 outputPerLane = 0;
        U64 pendingCapacity = 0;
        U64 workspaceSize = 0;
    } validatedPlan, plan;

    Tensor input;
    Tensor output;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_RATIONAL_RESAMPLER_MODULE_IMPL_HH
