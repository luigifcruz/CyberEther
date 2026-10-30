#ifndef JETSTREAM_DOMAINS_VISUALIZATION_EYE_DIAGRAM_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_EYE_DIAGRAM_MODULE_IMPL_HH

#include <vector>

#include <jetstream/domains/visualization/eye_diagram/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct EyeDiagramImpl : public Module::Impl, public DynamicConfig<EyeDiagram> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result reconfigure() override;

 protected:
    Tensor input;
    Tensor output;

    std::vector<F32> pending;
    U64 interval = 0;
    F32 previous = 0.0f;
    bool initialized = false;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_EYE_DIAGRAM_MODULE_IMPL_HH
