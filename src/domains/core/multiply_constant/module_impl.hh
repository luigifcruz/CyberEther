#ifndef JETSTREAM_DOMAINS_CORE_MULTIPLY_CONSTANT_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_MULTIPLY_CONSTANT_MODULE_IMPL_HH

#include <jetstream/domains/core/multiply_constant/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct MultiplyConstantImpl : public Module::Impl, public DynamicConfig<MultiplyConstant> {
 public:
    Result define() override;
    Result create() override;
    Result reconfigure() override;

 protected:
    Tensor input;
    Tensor output;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_MULTIPLY_CONSTANT_MODULE_IMPL_HH
