#ifndef JETSTREAM_DOMAINS_DSP_RRC_FILTER_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_RRC_FILTER_MODULE_IMPL_HH

#include <jetstream/domains/dsp/rrc_filter/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct RrcFilterImpl : public Module::Impl,
                       public DynamicConfig<RrcFilter> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result reconfigure() override;

 protected:
    Tensor input;
    Tensor output;
    Tensor coeffs;
    Tensor history;
    U64 historyIndex = 0;

    Result generateCoefficients();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_RRC_FILTER_MODULE_IMPL_HH
