#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_STACK_WINDOW_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_STACK_WINDOW_HH

#include "../components/hint_overlay.hh"

#include "jetstream/render/sakura/base.hh"

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphStackWindow {
    struct Config {
        std::string id;
        std::string title;
        Extent2D<F32> position;
        Extent2D<F32> size;
        std::optional<U64> parentDockId;
        bool dockIntoParent = false;
        bool restoreLayout = false;
        std::optional<Sakura::DockspaceWindow::DockLayout> layout;
        std::vector<Sakura::DockspaceWindow::DockableWindow> dockables;

        std::function<void(Extent2D<F32>, Extent2D<F32>)> onGeometry;
        std::function<void(std::optional<Sakura::DockspaceWindow::DockLayout>)> onLayout;
        std::function<void()> onClose;
    };

    void update(Config config) {
        this->config = std::move(config);
        dockspace.update({
            .id = this->config.id,
            .title = this->config.title,
            .position = this->config.position,
            .size = this->config.size,
            .parentDockId = this->config.parentDockId,
            .dockIntoParent = this->config.dockIntoParent,
            .restoreLayout = this->config.restoreLayout,
            .layout = this->config.layout,
            .dockables = this->config.dockables,
            .onGeometry = this->config.onGeometry,
            .onLayout = this->config.onLayout,
            .onClose = this->config.onClose,
        });
        hint.update({
            .id = this->config.id + ":empty-hint",
            .title = "Arrange Your Stack",
            .subtitle = "Drag surfaces into this stack to create your layout",
            .steps = {
                "Detach a visualization surface from the flowgraph.",
                "Drag the detached surface into this stack window.",
                "Resize and rearrange to build your layout.",
            },
            .hints = {
                "Drag a surface tab into this stack to dock it.",
                "Split the stack by dragging a surface to the edge.",
                "Stacks sync with your flowgraph.",
            },
            .size = {460.0f, 340.0f},
        });
    }

    void render(const Sakura::Context& ctx) {
        dockspace.render(ctx, [this](const Sakura::Context& ctx) {
            hint.render(ctx);
        });
    }

 private:
    Config config;
    Sakura::DockspaceWindow dockspace;
    HintOverlay hint;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_STACK_WINDOW_HH
