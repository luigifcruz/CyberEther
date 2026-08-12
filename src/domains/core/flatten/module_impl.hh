#ifndef JETSTREAM_DOMAINS_CORE_FLATTEN_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_FLATTEN_MODULE_IMPL_HH

#include <jetstream/detail/module_impl.hh>
#include <jetstream/domains/core/flatten/module.hh>

namespace Jetstream::Modules {

struct FlattenImpl : public Module::Impl, public DynamicConfig<Flatten> {
 public:
    Result define() override;
    Result create() override;

 protected:
    Tensor input;
    Tensor output;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_FLATTEN_MODULE_IMPL_HH
