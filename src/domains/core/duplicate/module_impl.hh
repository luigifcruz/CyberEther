#ifndef JETSTREAM_DOMAINS_CORE_DUPLICATE_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_DUPLICATE_MODULE_IMPL_HH

#include <jetstream/domains/core/duplicate/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct DuplicateImpl : public Module::Impl, public DynamicConfig<Duplicate> {
 public:
    Result define() override;
    Result create() override;

 protected:
    Tensor input;
    Tensor output;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_DUPLICATE_MODULE_IMPL_HH
