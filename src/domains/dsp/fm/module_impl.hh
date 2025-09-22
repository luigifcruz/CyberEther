#ifndef JETSTREAM_DOMAINS_DSP_FM_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_FM_MODULE_IMPL_HH

#include <jetstream/domains/dsp/fm/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct FmImpl : public Module::Impl, public DynamicConfig<FM> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    Tensor input;
    Tensor output;
    F32 kf = 0.0f;
    F32 ref = 0.0f;

    void updateCoefficients();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_FM_MODULE_IMPL_HH
