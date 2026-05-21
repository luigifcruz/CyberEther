#ifndef BLUEPRINT_GAIN_MODULE_IMPL_HH
#define BLUEPRINT_GAIN_MODULE_IMPL_HH

#include <blueprint/gain/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct BlueprintGainImpl : public Module::Impl, public DynamicConfig<BlueprintGain> {
  public:
    Result define() override;
    Result create() override;
    Result reconfigure() override;

  protected:
    Tensor input;
    Tensor output;
};

}  // namespace Jetstream::Modules

#endif  // BLUEPRINT_GAIN_MODULE_IMPL_HH
