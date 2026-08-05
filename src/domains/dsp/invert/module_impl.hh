#ifndef JETSTREAM_DOMAINS_DSP_INVERT_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_INVERT_MODULE_IMPL_HH

#include <jetstream/domains/dsp/invert/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct InvertImpl : public Module::Impl, public DynamicConfig<Invert> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor input;
    Tensor output;
    Index resolvedAxis = 0;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_INVERT_MODULE_IMPL_HH
