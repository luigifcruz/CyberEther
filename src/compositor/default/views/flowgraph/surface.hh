#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_SURFACE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_SURFACE_HH

#include "jetstream/render/sakura/base.hh"

#include "jetstream/surface.hh"

#include <functional>
#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphDetachedSurface {
    struct Config {
        std::string id;
        std::string title;
        Extent2D<F32> logicalSize = {512.0f, 512.0f};
        std::function<U64()> onResolveTexture;
        std::function<void(const Sakura::SurfaceResize&)> onSize;
        std::function<void(MouseEvent)> onMouse;
        std::function<void()> onClose;
    };

    void update(Config config) {
        this->config = std::move(config);
        window.update({
            .id = this->config.id,
            .title = this->config.title,
            .size = this->config.logicalSize,
            .padding = Extent2D<F32>{4.0f, 4.0f},
            .backgroundColor = ColorRGBA<F32>{0.0f, 0.0f, 0.0f, 1.0f},
            .onClose = this->config.onClose,
        });
        surface.update({
            .id = this->config.id + ":surface",
            .size = {0.0f, 0.0f},
            .onResolveTexture = this->config.onResolveTexture,
            .onSize = this->config.onSize,
            .onMouse = this->config.onMouse,
        });
    }

    void render(const Sakura::Context& ctx) {
        window.render(ctx, [this](const Sakura::Context& ctx) {
            surface.render(ctx);
        });
    }

 private:
    Config config;
    Sakura::Window window;
    Sakura::SurfaceView surface;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_SURFACE_HH
