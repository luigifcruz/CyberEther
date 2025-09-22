#ifndef JETSTREAM_DOMAINS_DSP_FOLD_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_FOLD_MODULE_IMPL_HH

#include <jetstream/domains/dsp/fold/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct FoldImpl : public Module::Impl, public DynamicConfig<Fold> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    Tensor input;
    Tensor output;
    U64 decimationFactor = 0;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_FOLD_MODULE_IMPL_HH
