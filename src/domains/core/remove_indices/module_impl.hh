#ifndef JETSTREAM_DOMAINS_CORE_REMOVE_INDICES_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_REMOVE_INDICES_MODULE_IMPL_HH

#include <jetstream/domains/core/remove_indices/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct RemoveIndicesImpl : public Module::Impl, public DynamicConfig<RemoveIndices> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    Index validatedResolvedAxis = 0;
    std::vector<U64> validatedRemovedIndices;

    Index resolvedAxis = 0;
    std::vector<U64> keptIndices;
    Tensor input;
    Tensor output;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_REMOVE_INDICES_MODULE_IMPL_HH
