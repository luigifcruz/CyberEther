#ifndef JETSTREAM_DOMAINS_CORE_COMPLEX_SPLIT_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_COMPLEX_SPLIT_MODULE_IMPL_HH

#include <jetstream/domains/core/complex_split/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct ComplexSplitImpl : public Module::Impl, public DynamicConfig<ComplexSplit> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    bool validatedPolar = false;

    Tensor input;
    Tensor outputA;
    Tensor outputB;
    bool polar = false;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_COMPLEX_SPLIT_MODULE_IMPL_HH
