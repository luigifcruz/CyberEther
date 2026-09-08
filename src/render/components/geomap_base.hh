#ifndef JETSTREAM_RENDER_COMPONENTS_GEOMAP_BASE_HH
#define JETSTREAM_RENDER_COMPONENTS_GEOMAP_BASE_HH

#include <array>
#include <memory>

#include "jetstream/render/base/surface.hh"
#include "jetstream/render/components/map_layer.hh"
#include "geomap_labels.hh"

namespace Jetstream::Render::Components {

// Built-in cartography. GeoMap itself only coordinates a stack of MapLayers;
// all Natural Earth policy and rendering resources stay in this base layer.
class GeoMapBaseLayer final : public MapLayer {
 public:
    using Uniforms = MapContext::Uniforms;
    using GpuUniforms = MapContext::GpuUniforms;

    struct LabelTuning {
        bool enabled = true;
        float zoomOffset = 0.0f;
        float fadeScale = 1.0f;
    };

    // Rendering policy: which layers draw and the detail zoom at which each
    // class of feature fades in or out. These are private cartographic defaults,
    // not part of the map host's extension API or persisted flowgraph parameters.
    struct Tuning : GeoMapLabels::Tuning {
        bool waterSphere = true;
        bool bathymetry = true;
        bool landcover = true;

        bool coastlines = true;
        float coastlineOpacity = 1.0f;
        bool countryBorders = true;
        float countryBorderOpacity = 0.72f;
        bool stateBorders = true;
        float stateLineFadeStart = GeoMapLabels::StateLineFadeStartZoom;
        float stateLineFull = GeoMapLabels::StateLineFullZoom;
        float stateLineOpacity = 0.75f;
        bool rivers = true;
        float riverZoomOffset = 0.0f;
        float riverFadeRange = 1.0f;
        bool references = true;
        std::array<float, 4> referenceMaxZoom = {5.5f, 5.5f, 4.5f, 5.5f};
        float referenceFadeRange = 1.5f;
        bool disputedBorders = true;
        float disputedZoomOffset = 0.0f;

        // Label order: country, capital, city, state,
        // region, marine, airport, physical/hydro.
        std::array<LabelTuning, 8> labels{};

        bool worldCityMarkers = true;
        bool capitalMarkers = true;
        bool cityMarkers = true;
        bool airportMarkers = true;

        bool atmosphere = true;
        float atmosphereRadius = 1.025f;
        float atmosphereInnerStrength = 0.20f;
        float atmosphereOuterStrength = 0.32f;
        float atmosphereOutlineStrength = 0.48f;
        std::array<float, 3> atmosphereColor = {0.25f, 0.52f, 0.95f};
    };

    GeoMapBaseLayer();
    ~GeoMapBaseLayer() override;

    Result create(Window* window, const MapContext& context) override;
    Result destroy(Window* window) override;
    Result surface(Render::Surface::Config& config) override;
    Result present(const MapContext& context) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

}  // namespace Jetstream::Render::Components

#endif  // JETSTREAM_RENDER_COMPONENTS_GEOMAP_BASE_HH
