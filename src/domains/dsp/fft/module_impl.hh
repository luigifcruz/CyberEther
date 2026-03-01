#ifndef JETSTREAM_DOMAINS_DSP_FFT_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_FFT_MODULE_IMPL_HH

#include <jetstream/domains/dsp/fft/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct FftImpl : public Module::Impl, public DynamicConfig<Fft> {
 public:
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor input;
    Tensor output;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_FFT_MODULE_IMPL_HH
