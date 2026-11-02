#ifndef JETSTREAM_DOMAINS_CORE_SIGNAL_ATTRIBUTES_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_SIGNAL_ATTRIBUTES_MODULE_IMPL_HH

#include <jetstream/domains/core/signal_attributes/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct SignalAttributesImpl : public Module::Impl,
                              public DynamicConfig<Modules::SignalAttributes> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    Tensor input;
    Tensor output;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_SIGNAL_ATTRIBUTES_MODULE_IMPL_HH
