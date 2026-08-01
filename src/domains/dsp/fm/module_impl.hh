#ifndef JETSTREAM_DOMAINS_DSP_FM_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_FM_MODULE_IMPL_HH

#include <jetstream/domains/dsp/fm/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/memory/axis.hh>

#include <vector>

namespace Jetstream::Modules {

struct FmImpl : public Module::Impl, public DynamicConfig<FM> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    Tensor input;
    Tensor output;
    F32 kf = 0.0f;
    F32 ref = 0.0f;

    SignalAxes validatedSignalAxes;
    U64 validatedLaneCount = 0;

    SignalAxes signalAxes;
    U64 laneCount = 0;
    std::vector<CF32> previousSample;
    std::vector<U8> hasPreviousSample;

    void updateCoefficients();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_FM_MODULE_IMPL_HH
