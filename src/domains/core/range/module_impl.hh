#ifndef JETSTREAM_DOMAINS_CORE_RANGE_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_RANGE_MODULE_IMPL_HH

#include <jetstream/domains/core/range/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct RangeImpl : public Module::Impl, public DynamicConfig<Range> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor input;
    Tensor output;

    F32 scalingCoeff;
    F32 offsetCoeff;

    void updateCoefficients();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_RANGE_MODULE_IMPL_HH
