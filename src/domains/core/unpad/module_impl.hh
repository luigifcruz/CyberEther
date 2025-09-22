#ifndef JETSTREAM_DOMAINS_CORE_UNPAD_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_UNPAD_MODULE_IMPL_HH

#include <jetstream/domains/core/unpad/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct UnpadImpl : public Module::Impl, public DynamicConfig<Unpad> {
 public:
    Result define() override;
    Result create() override;

 protected:
    Tensor input;
    Tensor outputUnpadded;
    Tensor outputPad;

    U64 inputAxisSize = 0;
    U64 unpadAxisSize = 0;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_UNPAD_MODULE_IMPL_HH
