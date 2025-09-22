#ifndef JETSTREAM_DOMAINS_DSP_AGC_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_AGC_MODULE_IMPL_HH

#include <jetstream/domains/dsp/agc/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct AgcImpl : public Module::Impl, public DynamicConfig<Agc> {
 public:
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor input;
    Tensor output;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_AGC_MODULE_IMPL_HH
