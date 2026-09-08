#ifndef JETSTREAM_RENDER_COMPONENTS_MAP_LAYER_HH
#define JETSTREAM_RENDER_COMPONENTS_MAP_LAYER_HH

#include <memory>
#include <vector>

#include "jetstream/render/components/map_context.hh"
#include "jetstream/render/base/surface.hh"

namespace Jetstream::Render { class Window; }

namespace Jetstream::Render::Components {

class JETSTREAM_API MapLayer {
 public:
    enum class EventResult { Ignored, Handled, Capture };

    virtual ~MapLayer() = default;
    virtual Result create(Window*, const MapContext&) { return Result::SUCCESS; }
    virtual Result destroy(Window*) { return Result::SUCCESS; }
    virtual Result surface(Render::Surface::Config&) = 0;
    virtual Result present(const MapContext&) = 0;
    virtual EventResult onEvent(const MouseEvent&, const MapContext&) {
        return EventResult::Ignored;
    }
};

class JETSTREAM_API MapLayerStack {
 public:
    MapLayerStack() = default;
    MapLayerStack(const MapLayerStack&) = delete;
    MapLayerStack& operator=(const MapLayerStack&) = delete;

    Result add(const std::shared_ptr<MapLayer>& layer);
    Result create(Window* window, const MapContext& context);
    Result destroy(Window* window);
    Result surface(Render::Surface::Config& config);
    Result present(const MapContext& context);
    bool onEvent(const MouseEvent& event, const MapContext& context);

 private:
    std::vector<std::shared_ptr<MapLayer>> layers;
    std::shared_ptr<MapLayer> captured;
    U64 createdCount = 0;
    bool active = false;
};

}  // namespace Jetstream::Render::Components

#endif
