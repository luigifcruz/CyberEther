#ifndef JETSTREAM_DOMAINS_DSP_OVERLAP_ADD_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_OVERLAP_ADD_MODULE_IMPL_HH

#include <jetstream/domains/dsp/overlap_add/module.hh>
#include <jetstream/detail/module_impl.hh>

#include <optional>

namespace Jetstream::Modules {

struct OverlapAddImpl : public Module::Impl,
                        public DynamicConfig<OverlapAdd> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;

 protected:
    Tensor validatedInputBuffer;
    Tensor validatedInputOverlap;
    std::optional<Index> validatedBatchAxis;
    Shape validatedPreviousOverlapShape;
    U64 validatedOutputSizeBytes = 0;
    U64 validatedPreviousOverlapSizeBytes = 0;

    Tensor inputBuffer;
    Tensor inputOverlap;
    Tensor output;
    Tensor previousOverlap;
    std::optional<Index> batchAxis;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_OVERLAP_ADD_MODULE_IMPL_HH
