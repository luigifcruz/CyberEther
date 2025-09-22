#ifndef JETSTREAM_DOMAINS_DSP_AM_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_AM_MODULE_IMPL_HH

#include <jetstream/domains/dsp/am/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct AmImpl : public Module::Impl, public DynamicConfig<AM> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    Tensor input;
    Tensor output;

    F32 prevEnvelope = 0.0f;
    F32 prevOutput = 0.0f;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_AM_MODULE_IMPL_HH
