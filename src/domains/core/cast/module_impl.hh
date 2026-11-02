#ifndef JETSTREAM_DOMAINS_CORE_CAST_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_CAST_MODULE_IMPL_HH

#include <jetstream/domains/core/cast/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct CastImpl : public Module::Impl, public DynamicConfig<Cast> {
 public:
    struct Scale {
        bool integer = false;
        bool isUnsigned = false;
        U64 bits = 0;
        F64 fullScale = 1.0;
    };

    static Scale ScaleOf(const DataType& dtype);

    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    DataType validatedOutputDtype = DataType::None;
    bool validatedBypass = false;
    U64 validatedOutputElementCount = 0;
    U64 validatedOutputSizeBytes = 0;

    Tensor input;
    Tensor output;
    DataType outputDtype = DataType::CF32;
    Scale inputScale;
    Scale outputScale;
    bool bypass = false;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_CAST_MODULE_IMPL_HH
