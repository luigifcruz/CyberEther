#ifndef JETSTREAM_DOMAINS_DSP_AGC_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_AGC_MODULE_IMPL_HH

#include <jetstream/domains/dsp/agc/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/memory/axis.hh>

namespace Jetstream::Modules {

struct AgcImpl : public Module::Impl, public DynamicConfig<Agc> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor input;
    Tensor output;
    SignalAxes validatedSignalAxes;
    Index sampleAxis = 0;
    U64 laneCount = 0;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_AGC_MODULE_IMPL_HH
