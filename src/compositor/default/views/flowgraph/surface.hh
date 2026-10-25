#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_SURFACE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_SURFACE_HH

#include "jetstream/render/sakura/base.hh"

#include "jetstream/render/tools/icons_lucide.hh"
#include "jetstream/surface.hh"

#include "editor/config/base.hh"

#include <functional>
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
        std::function<void(const Sakura::SurfaceResize&)> onSize;
        std::function<void(InputEvent)> onInput;
        std::function<SurfaceCursor()> resolveCursor;
        std::function<void()> onClose;
        std::function<void(bool)> onToggleConfigOpen;
    };

    void update(Config config) {
        if (config.id != this->config.id || !initialized) {
            configVisible = config.configOpen;
            initialized = true;
        }
        this->config = std::move(config);
        surface.update({
            .texture = this->config.texture,
            .onSize = this->config.onSize,
            .onInput = this->config.onInput,
            .resolveCursor = this->config.resolveCursor,
        });

        while (fields.size() > this->config.configFields.size()) {
            fields.pop_back();
        }
        while (fields.size() < this->config.configFields.size()) {
            fields.push_back(std::make_unique<FlowgraphConfigFieldInstance>());
        }
        std::vector<Sakura::NodeFieldGrid::Item> items;
        items.reserve(fields.size());
        for (U64 i = 0; i < fields.size(); ++i) {
            fields[i]->update(this->config.configFields[i]);
            items.push_back({
                .child = &fields[i]->component(),
                .fullWidth = !fields[i]->isSimple(),
            });
        }
        fieldGrid.update({
            .id = this->config.id + "FieldGrid",
            .items = std::move(items),
        });
        fieldStack.update({
            .id = this->config.id + ":fields",
            .items = {{.child = &fieldGrid}},
        });
        fieldPanel.update({
            .id = this->config.id + ":fields",
            .content = &fieldStack,
        });
        std::vector<Sakura::Stack::Item> rows;
        if (!fields.empty() && configVisible) {
            rows.push_back({.child = &fieldPanel});
        }
        rows.push_back({.child = &surface, .grow = 1.0f});
        layout.update({.items = std::move(rows)});
    }

    std::vector<Sakura::Dock::Action> actions() {
        if (fields.empty()) {
            return {};
        }
        return {{
            .icon = ICON_LC_SLIDERS_HORIZONTAL,
            .active = configVisible,
            .onClick = [this]() {
                configVisible = !configVisible;
                if (config.onToggleConfigOpen) {
                    config.onToggleConfigOpen(configVisible);
                }
            },
        }};
    }

    Sakura::Component& component() {
        return layout;
    }

    const std::string& title() const {
        return config.title;
    }

    std::function<void()> closeCallback() const {
        return config.onClose;
    }

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
