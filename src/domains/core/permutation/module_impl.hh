#ifndef JETSTREAM_DOMAINS_CORE_PERMUTATION_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_PERMUTATION_MODULE_IMPL_HH

#include <jetstream/domains/core/permutation/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct PermutationImpl : public Module::Impl, public DynamicConfig<Permutation> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    Tensor input;
    Tensor output;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_PERMUTATION_MODULE_IMPL_HH
