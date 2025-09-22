#ifndef JETSTREAM_DOMAINS_DSP_OVERLAP_ADD_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_OVERLAP_ADD_MODULE_IMPL_HH

#include <jetstream/domains/dsp/overlap_add/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct OverlapAddImpl : public Module::Impl,
                        public DynamicConfig<OverlapAdd> {
 public:
    Result define() override;
    Result create() override;
    Result destroy() override;

 protected:
    Tensor inputBuffer;
    Tensor inputOverlap;
    Tensor output;
    Tensor previousOverlap;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_OVERLAP_ADD_MODULE_IMPL_HH
