#ifndef JETSTREAM_DOMAINS_CORE_COMPLEX_MERGE_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_COMPLEX_MERGE_MODULE_IMPL_HH

#include <jetstream/domains/core/complex_merge/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct ComplexMergeImpl : public Module::Impl, public DynamicConfig<ComplexMerge> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    bool validatedPolar = false;

    Tensor inputA;
    Tensor inputB;
    Tensor output;
    bool polar = false;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_COMPLEX_MERGE_MODULE_IMPL_HH
