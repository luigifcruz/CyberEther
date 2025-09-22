#ifndef JETSTREAM_DOMAINS_DSP_FILTER_TAPS_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_FILTER_TAPS_MODULE_IMPL_HH

#include <jetstream/domains/dsp/filter_taps/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct FilterTapsImpl : public Module::Impl, public DynamicConfig<FilterTaps> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;

 protected:
    Tensor coeffs;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_FILTER_TAPS_MODULE_IMPL_HH
