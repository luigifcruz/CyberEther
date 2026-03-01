#ifndef JETSTREAM_DOMAINS_CORE_RESHAPE_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_CORE_RESHAPE_MODULE_IMPL_HH

#include <regex>

#include <jetstream/domains/core/reshape/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct ReshapeImpl : public Module::Impl, public DynamicConfig<Reshape> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;

 protected:
    Tensor input;
    Tensor output;

    Shape parsedShape;

    Result parseShapeString(const std::string& shapeStr, Shape& parsedShape);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_RESHAPE_MODULE_IMPL_HH
