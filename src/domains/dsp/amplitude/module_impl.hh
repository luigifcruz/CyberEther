#ifndef JETSTREAM_DOMAINS_DSP_AMPLITUDE_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_AMPLITUDE_MODULE_IMPL_HH

#include <jetstream/domains/dsp/amplitude/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct AmplitudeImpl : public Module::Impl, public DynamicConfig<Amplitude> {
 public:
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor input;
    Tensor output;
    F32 scalingCoeff = 0.0f;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_AMPLITUDE_MODULE_IMPL_HH
