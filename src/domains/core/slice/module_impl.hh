#ifndef JETSTREAM_DOMAINS_CORE_SLICE_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_SLICE_MODULE_IMPL_HH

#include <jetstream/domains/core/slice/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct SliceImpl : public Module::Impl, public DynamicConfig<Slice> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    Tensor input;
    Tensor output;

    std::vector<Token> sliceTokens;
    Tensor::SlicePlan slicePlan;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_SLICE_MODULE_IMPL_HH
