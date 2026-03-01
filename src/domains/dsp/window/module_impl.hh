#ifndef JETSTREAM_DOMAINS_DSP_WINDOW_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_WINDOW_MODULE_IMPL_HH

#include <jetstream/domains/dsp/window/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct WindowImpl : public Module::Impl, public DynamicConfig<Window> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    Tensor output;
    bool baked = false;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_WINDOW_MODULE_IMPL_HH
