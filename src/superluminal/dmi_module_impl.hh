#ifndef JETSTREAM_SUPERLUMINAL_DMI_MODULE_IMPL_HH
#define JETSTREAM_SUPERLUMINAL_DMI_MODULE_IMPL_HH

#include "dmi_module.hh"
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct DynamicTensorImportImpl : public Module::Impl,
                                 public DynamicConfig<DynamicTensorImport> {
 public:
    Result define() override;
    Result create() override;

 protected:
    Tensor output;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_SUPERLUMINAL_DMI_MODULE_IMPL_HH
