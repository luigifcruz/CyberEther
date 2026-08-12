#ifndef JETSTREAM_DOMAINS_CORE_PAD_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_PAD_MODULE_IMPL_HH

#include <jetstream/domains/core/pad/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct PadImpl : public Module::Impl, public DynamicConfig<Pad> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    Tensor input;
    Tensor output;

    Index validatedResolvedAxis = 0;
    U64 validatedInputAxisSize = 0;
    U64 validatedOutputAxisSize = 0;
    U64 validatedOutputSizeBytes = 0;

    Index resolvedAxis = 0;
    U64 inputAxisSize = 0;
    U64 outputAxisSize = 0;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_PAD_MODULE_IMPL_HH
