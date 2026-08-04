#ifndef JETSTREAM_DOMAINS_DSP_AM_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_AM_MODULE_IMPL_HH

#include <jetstream/domains/dsp/am/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/memory/axis.hh>

#include <vector>

namespace Jetstream::Modules {

struct AmImpl : public Module::Impl, public DynamicConfig<AM> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    Tensor input;
    Tensor output;

    SignalAxes validatedSignalAxes;
    U64 validatedLaneCount = 0;

    SignalAxes signalAxes;
    U64 laneCount = 0;
    std::vector<F32> prevEnvelope;
    std::vector<F32> prevOutput;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_AM_MODULE_IMPL_HH
