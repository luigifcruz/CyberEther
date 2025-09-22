#ifndef JETSTREAM_DOMAINS_CORE_PAD_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_PAD_MODULE_IMPL_HH

#include <jetstream/domains/core/pad/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct PadImpl : public Module::Impl, public DynamicConfig<Pad> {
 public:
    Result define() override;
    Result create() override;

 protected:
    Tensor input;
    Tensor output;

    U64 inputAxisSize = 0;
    U64 outputAxisSize = 0;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_PAD_MODULE_IMPL_HH
