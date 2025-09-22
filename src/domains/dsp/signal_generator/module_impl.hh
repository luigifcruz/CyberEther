#ifndef JETSTREAM_DOMAINS_DSP_SIGNAL_GENERATOR_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_SIGNAL_GENERATOR_MODULE_IMPL_HH

#include <jetstream/domains/dsp/signal_generator/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct SignalGeneratorImpl : public Module::Impl, public DynamicConfig<SignalGenerator> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor signal;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_SIGNAL_GENERATOR_MODULE_IMPL_HH
