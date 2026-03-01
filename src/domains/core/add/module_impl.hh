#ifndef JETSTREAM_DOMAINS_CORE_ADD_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_ADD_MODULE_IMPL_HH

#include <jetstream/domains/core/add/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct AddImpl : public Module::Impl, public DynamicConfig<Add> {
 public:
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor a;
    Tensor b;
    Tensor c;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_ADD_MODULE_IMPL_HH
