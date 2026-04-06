#ifndef JETSTREAM_DOMAINS_CORE_ONES_TENSOR_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_ONES_TENSOR_MODULE_IMPL_HH

#include <jetstream/detail/module_impl.hh>
#include <jetstream/domains/core/ones_tensor/module.hh>

namespace Jetstream::Modules {

struct OnesTensorImpl : public Module::Impl, public DynamicConfig<OnesTensor> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    Tensor output;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_ONES_TENSOR_MODULE_IMPL_HH
