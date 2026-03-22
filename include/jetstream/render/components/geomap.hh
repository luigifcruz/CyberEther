#ifndef JETSTREAM_RENDER_COMPONENTS_GEOMAP_HH
#define JETSTREAM_RENDER_COMPONENTS_GEOMAP_HH

#include <memory>
#include <vector>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"

#include "jetstream/render/base/surface.hh"
#include "jetstream/render/components/generic.hh"

namespace Jetstream::Render::Components {

class GeoMap : public Generic {
 public:
    struct Config {
    };

    struct Uniforms {
        float centerLon = 0.0f;
        float centerLat = 0.0f;
        float zoom = 1.0f;
        float aspectRatio = 1.0f;
        float viewportWidth = 800.0f;
        float viewportHeight = 600.0f;
    };

    GeoMap(const Config& config);
    ~GeoMap();

    Result create(Window* window);
    Result destroy(Window* window);

    Result surface(Render::Surface::Config& config);

    Result present();

    Result updateUniforms(const Uniforms& uniforms);

    constexpr const Config& getConfig() const {
        return config;
    }

    const Uniforms& getUniforms() const;

 private:
    Config config;

    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

}  // namespace Jetstream::Render::Components

#endif  // JETSTREAM_RENDER_COMPONENTS_GEOMAP_HH
