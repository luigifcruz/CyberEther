#ifndef JETSTREAM_DOMAINS_CORE_MULTIPLY_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_MULTIPLY_MODULE_IMPL_HH

#include <jetstream/domains/core/multiply/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct MultiplyImpl : public Module::Impl, public DynamicConfig<Multiply> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor validatedA;
    Tensor validatedB;
    Shape validatedOutputShape;
    U64 validatedOutputElementCount = 0;
    U64 validatedOutputSizeBytes = 0;

    Tensor a;
    Tensor b;
    Tensor c;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_MULTIPLY_MODULE_IMPL_HH
