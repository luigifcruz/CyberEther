#ifndef JETSTREAM_DOMAINS_VISUALIZATION_CONSTELLATION_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_CONSTELLATION_MODULE_IMPL_HH

#include <jetstream/domains/visualization/constellation/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/surface.hh>
#include <jetstream/render/base/texture.hh>
#include <jetstream/render/base/surface.hh>
#include <jetstream/render/components/axis.hh>
#include <jetstream/render/components/shapes.hh>

namespace Jetstream::Modules {

constexpr F32 kConstellationPointSize = 10.0f;
constexpr F32 kConstellationZoomSpeed = 0.15f;

struct ConstellationImpl : public Module::Impl,
                           public DynamicConfig<Constellation> {
 public:
    Result define() override;
    Result create() override;
    Result destroy() override;

 protected:
    Tensor input;

    U64 numberOfPoints = 0;

    // Surface interaction state.
    SurfaceInteractionState interaction;

    // Rendering state.
    std::shared_ptr<Render::Texture> framebufferTexture;
    std::shared_ptr<Render::Surface> renderSurface;
    std::shared_ptr<Render::Components::Axis> axis;
    std::shared_ptr<Render::Components::Shapes> shapes;

    // Update flags.
    bool updatePositionsFlag = false;

    Result createPresent();
    Result destroyPresent();
    Result present();
    Result updateAxisState();
    Result updatePointPositions();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_CONSTELLATION_MODULE_IMPL_HH
