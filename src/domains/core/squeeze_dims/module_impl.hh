#ifndef JETSTREAM_DOMAINS_CORE_SQUEEZE_DIMS_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_SQUEEZE_DIMS_MODULE_IMPL_HH

#include <jetstream/domains/core/squeeze_dims/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct SqueezeDimsImpl : public Module::Impl, public DynamicConfig<SqueezeDims> {
 public:
    Result define() override;
    Result create() override;

 protected:
    Tensor input;
    Tensor output;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_SQUEEZE_DIMS_MODULE_IMPL_HH
