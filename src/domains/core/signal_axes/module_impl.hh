#ifndef JETSTREAM_DOMAINS_CORE_SIGNAL_AXES_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_SIGNAL_AXES_MODULE_IMPL_HH

#include <jetstream/domains/core/signal_axes/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/memory/axis.hh>

namespace Jetstream::Modules {

struct SignalAxesImpl : public Module::Impl,
                        public DynamicConfig<Modules::SignalAxes> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    Tensor input;
    Tensor output;
    bool validatedOverrideAxes = false;
    Jetstream::SignalAxes validatedAxes;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_SIGNAL_AXES_MODULE_IMPL_HH
