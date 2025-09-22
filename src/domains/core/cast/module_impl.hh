#ifndef JETSTREAM_DOMAINS_CORE_CAST_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_CAST_MODULE_IMPL_HH

#include <jetstream/domains/core/cast/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct CastImpl : public Module::Impl, public DynamicConfig<Cast> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    Tensor input;
    Tensor output;
    DataType outputDtype = DataType::CF32;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_CAST_MODULE_IMPL_HH
