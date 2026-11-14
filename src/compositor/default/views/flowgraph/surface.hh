#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_SURFACE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_SURFACE_HH

#include "jetstream/render/sakura/base.hh"

#include "jetstream/render/tools/icons_lucide.hh"
#include "jetstream/surface.hh"

#include "editor/config/base.hh"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphDetachedSurface {
    struct Config {
        std::string id;
        std::string title;
        std::vector<FlowgraphConfigFieldConfig> configFields;
        bool configOpen = false;
        std::shared_ptr<const Render::Texture> texture;
        Sakura::Callback<void(const Sakura::SurfaceResize&)> onSize;
        Sakura::Callback<void(InputEvent)> onInput;
        Sakura::Callback<SurfaceCursor()> resolveCursor;
        Sakura::Callback<void()> onClose;
        Sakura::Callback<void(bool)> onToggleConfigOpen;
    };

    void update(Config config);
    std::vector<Sakura::Dock::Action> actions();
    Sakura::Component& component();
    const std::string& title() const;
    Sakura::Callback<void()> closeCallback() const;

 private:
    Config config;
    bool configVisible = false;
    bool initialized = false;
    Sakura::NodeFieldGrid fieldGrid;
    Sakura::NodeContentStack fieldStack;
    Sakura::NodeFieldPanel fieldPanel;
    std::vector<std::unique_ptr<FlowgraphConfigFieldInstance>> fields;
    Sakura::SurfaceImage surface;
    Sakura::Stack layout;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_SURFACE_HH
