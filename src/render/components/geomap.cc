#include "jetstream/render/components/geomap.hh"

#include <array>
#include <atomic>
#include <cmath>

#include "geomap_base.hh"

namespace Jetstream::Render::Components {

namespace {

constexpr std::array RenderingFields = {
    &MapRenderingOptions::buildings, &MapRenderingOptions::buildingOutlines,
    &MapRenderingOptions::roads, &MapRenderingOptions::rivers,
    &MapRenderingOptions::borders, &MapRenderingOptions::places,
    &MapRenderingOptions::airports, &MapRenderingOptions::bathymetry,
};
std::atomic<U16> RenderingFlags{(1u << RenderingFields.size()) - 1};

}  // namespace

void GeoMap::SetRenderingOptions(const MapRenderingOptions& options) {
    U16 flags = 0;
    for (U64 i = 0; i < RenderingFields.size(); ++i) {
        if (options.*RenderingFields[i]) flags |= 1u << i;
    }
    RenderingFlags.store(flags, std::memory_order_relaxed);
}

MapRenderingOptions GeoMap::RenderingOptions() {
    const U16 flags = RenderingFlags.load(std::memory_order_relaxed);
    MapRenderingOptions options;
    for (U64 i = 0; i < RenderingFields.size(); ++i) {
        options.*RenderingFields[i] = (flags & (1u << i)) != 0;
    }
    return options;
}

struct GeoMap::Impl {
    MapContext context;
    MapNavigation navigation;
    MapLayerStack layers;
};

GeoMap::GeoMap(const Config& config)
    : config(config), pimpl(std::make_unique<Impl>()) {
    pimpl->layers.add(std::make_shared<GeoMapBaseLayer>());
}

GeoMap::~GeoMap() = default;

Result GeoMap::addLayer(const std::shared_ptr<MapLayer>& layer) {
    return pimpl->layers.add(layer);
}

Result GeoMap::create(Window* window) {
    if (!window) return Result::ERROR;
    return pimpl->layers.create(window, pimpl->context);
}

Result GeoMap::destroy(Window* window) {
    pimpl->navigation.cancel();
    pimpl->context.cursor.reset();
    return pimpl->layers.destroy(window);
}

Result GeoMap::surface(Render::Surface::Config& config) {
    return pimpl->layers.surface(config);
}

Result GeoMap::present() {
    return pimpl->layers.present(pimpl->context);
}

Result GeoMap::updateUniforms(const Uniforms& uniforms) {
    return pimpl->context.update(uniforms);
}

Result GeoMap::processInteraction(std::span<const SurfaceEvent> surfaceEvents,
                                  std::span<const MouseEvent> mouseEvents) {
    auto& context = pimpl->context;
    for (const auto& event : surfaceEvents) {
        JST_CHECK(pimpl->navigation.resize(event, context));
    }
    for (const auto& event : mouseEvents) {
        if (event.type == MouseEventType::Leave) {
            context.cursor.reset();
        } else {
            if (!std::isfinite(event.position.x) ||
                !std::isfinite(event.position.y)) continue;
            context.cursor = event.position;
        }
        if (pimpl->layers.onEvent(event, context)) {
            // An overlay taking the pointer must not leave a globe drag active.
            pimpl->navigation.cancel();
        } else {
            JST_CHECK(pimpl->navigation.mouse(event, context));
        }
    }
    return Result::SUCCESS;
}

const MapContext& GeoMap::getContext() const { return pimpl->context; }
const GeoMap::Uniforms& GeoMap::getUniforms() const { return pimpl->context.view; }

}  // namespace Jetstream::Render::Components
