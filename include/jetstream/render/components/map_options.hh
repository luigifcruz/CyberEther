#ifndef JETSTREAM_RENDER_COMPONENTS_MAP_OPTIONS_HH
#define JETSTREAM_RENDER_COMPONENTS_MAP_OPTIONS_HH

#include "jetstream/parser.hh"

namespace Jetstream::Render::Components {

struct MapRenderingOptions {
    bool buildings = true;
    bool buildingOutlines = true;
    bool roads = true;
    bool rivers = true;
    bool borders = true;
    bool places = true;
    bool airports = true;
    bool bathymetry = true;

    bool operator==(const MapRenderingOptions&) const = default;

    JST_SERDES(buildings, buildingOutlines, roads, rivers, borders, places, airports, bathymetry);
};

}  // namespace Jetstream::Render::Components

#endif  // JETSTREAM_RENDER_COMPONENTS_MAP_OPTIONS_HH
