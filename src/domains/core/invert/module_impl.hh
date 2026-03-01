#ifndef JETSTREAM_DOMAINS_CORE_INVERT_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_INVERT_MODULE_IMPL_HH

#include <jetstream/domains/core/invert/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct InvertImpl : public Module::Impl, public DynamicConfig<Invert> {
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

#endif  // JETSTREAM_DOMAINS_CORE_INVERT_MODULE_IMPL_HH
