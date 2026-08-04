#ifndef JETSTREAM_DOMAINS_DSP_PHASE_CORRECTION_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_PHASE_CORRECTION_MODULE_IMPL_HH

#include <optional>
#include <vector>

#include <jetstream/detail/module_impl.hh>
#include <jetstream/domains/dsp/phase_correction/module.hh>

namespace Jetstream::Modules {

struct PhaseCorrectionImpl : public Module::Impl,
                             public DynamicConfig<PhaseCorrection> {
    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    std::optional<Index> validatedBatchAxis;
    std::optional<Index> validatedChannelAxis;
    std::vector<F64> validatedChannelPhaseIncrements;
    Tensor input;
    Tensor output;
    std::optional<Index> batchAxis;
    std::optional<Index> channelAxis;
    std::vector<F64> channelPhaseIncrements;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_PHASE_CORRECTION_MODULE_IMPL_HH
