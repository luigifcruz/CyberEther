#ifndef JETSTREAM_RENDER_COMPONENTS_GEOMAP_HH
#define JETSTREAM_RENDER_COMPONENTS_GEOMAP_HH

#include <memory>
#include <span>

#include "jetstream/render/components/generic.hh"
#include "jetstream/render/components/map_layer.hh"

namespace Jetstream::Render::Components {

class JETSTREAM_API GeoMap : public Generic {
 public:
    struct Config {};
    using Uniforms = MapContext::Uniforms;

    explicit GeoMap(const Config& config);
    ~GeoMap() override;

    Result addLayer(const std::shared_ptr<MapLayer>& layer);

    Result create(Window* window) override;
    Result destroy(Window* window) override;
    Result surface(Render::Surface::Config& config);
    Result present() override;

    Result updateUniforms(const Uniforms& uniforms);
    Result processInteraction(std::span<const SurfaceEvent> surfaceEvents,
                              std::span<const MouseEvent> mouseEvents);

    const MapContext& getContext() const;
    const Uniforms& getUniforms() const;

    constexpr const Config& getConfig() const { return config; }

 private:
    Config config;
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

}  // namespace Jetstream::Render::Components

#endif
