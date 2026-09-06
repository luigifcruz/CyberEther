#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_SURFACE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_SURFACE_HH

#include "jetstream/render/sakura/base.hh"

#include "jetstream/surface.hh"

#include "editor/config/base.hh"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphDetachedSurface {
    struct Config {
        std::string id;
        std::string title;
        Extent2D<F32> logicalSize = {512.0f, 512.0f};
        std::vector<FlowgraphConfigFieldConfig> configFields;
        bool configOpen = false;
        std::function<U64()> onResolveTexture;
        std::function<void(const Sakura::SurfaceResize&)> onSize;
        std::function<void(MouseEvent)> onMouse;
        std::function<void()> onClose;
        std::function<void(bool)> onToggleConfigOpen;
    };

    void update(Config config) {
        if (config.id != this->config.id || !initialized) {
            configVisible = config.configOpen;
            initialized = true;
        }
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

        fields.resize(this->config.configFields.size());
        for (U64 i = 0; i < fields.size(); ++i) {
            fields[i].update(this->config.configFields[i]);
        }
        fieldGrid.update({
            .id = this->config.id + "FieldGrid",
        });
        chevron.update({
            .id = this->config.id + ":chevron",
        });
    }

    void render(const Sakura::Context& ctx) {
        window.render(ctx, [this](const Sakura::Context& ctx) {
            if (!fields.empty()) {
                if (chevron.render(ctx, configVisible)) {
                    configVisible = !configVisible;
                    if (config.onToggleConfigOpen) {
                        config.onToggleConfigOpen(configVisible);
                    }
                }

                if (configVisible) {
                    std::vector<Sakura::NodeFieldGrid::Item> items;
                    items.reserve(fields.size());
                    for (U64 i = 0; i < fields.size(); ++i) {
                        items.push_back({
                            .child = [this, i](const Sakura::Context& ctx) {
                                fields[i].render(ctx);
                            },
                            .fullWidth = !fields[i].isSimple(),
                        });
                    }
                    fieldGrid.render(ctx, items);
                }
            }
            surface.render(ctx);
        });
    }

 private:
    Config config;
    bool configVisible = false;
    bool initialized = false;
    Sakura::Window window;
    Sakura::CollapseChevron chevron;
    Sakura::NodeFieldGrid fieldGrid;
    std::vector<FlowgraphConfigFieldInstance> fields;
    Sakura::SurfaceView surface;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_SURFACE_HH
