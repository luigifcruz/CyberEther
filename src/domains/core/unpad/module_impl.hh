#ifndef JETSTREAM_DOMAINS_CORE_UNPAD_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_UNPAD_MODULE_IMPL_HH

#include <jetstream/domains/core/unpad/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct UnpadImpl : public Module::Impl, public DynamicConfig<Unpad> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    Tensor input;
    Tensor outputUnpadded;
    Tensor outputPad;

    Index validatedResolvedAxis = 0;
    U64 validatedInputAxisSize = 0;
    U64 validatedUnpadAxisSize = 0;

    Index resolvedAxis = 0;
    U64 inputAxisSize = 0;
    U64 unpadAxisSize = 0;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_UNPAD_MODULE_IMPL_HH
