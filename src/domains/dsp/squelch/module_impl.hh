#ifndef JETSTREAM_DOMAINS_DSP_SQUELCH_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_SQUELCH_MODULE_IMPL_HH

#include <jetstream/domains/dsp/squelch/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/snapshot.hh>

namespace Jetstream::Modules {

struct SquelchImpl : public Module::Impl, public DynamicConfig<Squelch> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

    bool getPassing() const;
    F32 getAmplitude() const;

  protected:
    Tensor input;
    Tensor output;
    Tools::Snapshot<bool> passingState{false};
    Tools::Snapshot<F32> amplitudeState{0.0f};
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_SQUELCH_MODULE_IMPL_HH
