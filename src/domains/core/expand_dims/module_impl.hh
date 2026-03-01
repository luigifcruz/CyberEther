#ifndef JETSTREAM_DOMAINS_CORE_EXPAND_DIMS_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_EXPAND_DIMS_MODULE_IMPL_HH

#include <jetstream/domains/core/expand_dims/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct ExpandDimsImpl : public Module::Impl, public DynamicConfig<ExpandDims> {
 public:
    Result define() override;
    Result create() override;

 protected:
    Tensor input;
    Tensor output;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_EXPAND_DIMS_MODULE_IMPL_HH
