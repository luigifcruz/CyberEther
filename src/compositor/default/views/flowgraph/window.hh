#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_WINDOW_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_WINDOW_HH

#include "../components/hint_overlay.hh"
#include "editor/base.hh"
#include "surface.hh"
#include "stack.hh"
#include "toolbar.hh"

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphWindow : public Sakura::Component {
    struct Config {
        std::string id;
        std::string title;
        std::optional<U64> dockId;
        FlowgraphEditor::Config editor;
        std::vector<FlowgraphStackWindow::Config> stacks;
        std::vector<FlowgraphDetachedSurface::Config> detachedSurfaces;
        bool empty = false;
        std::function<void()> onSave;
        std::function<void()> onClose;
        std::function<void()> onCreateStack;
    };

    void update(Config config) {
        auto editorConfig = std::move(config.editor);
        editorConfig.openBlockPicker = openBlockPickerRequest;
        openBlockPickerRequest = false;
        this->config = std::move(config);

        const std::vector<std::string> flowgraphHints = {
            "Drag from output to input to connect blocks.",
            "Drag blocks to move them around.",
            "Right-click a block to open its context menu.",
            "Use Ctrl+C and Ctrl+V to copy and paste blocks.",
            "Hover over ports to see their data type.",
            "Click and drag on empty canvas to pan around.",
        };

        const std::string title = this->config.title.empty() ? this->config.id : this->config.title;
        window.update({
            .id = this->config.id,
            .title = title,
            .size = {640.0f, 480.0f},
            .dockId = this->config.dockId,
            .padding = Extent2D<F32>{8.0f, 8.0f},
            .onClose = this->config.onClose,
        });
        toolbar.update({
            .id = this->config.id + ":toolbar",
            .onSave = this->config.onSave,
            .onClose = this->config.onClose,
            .onAddBlock = [this]() {
                openBlockPickerRequest = true;
            },
            .onCreateStack = this->config.onCreateStack,
        });
        editor.update(std::move(editorConfig));

        // Update stack windows.

        stackOrder.clear();
        std::unordered_set<std::string> activeStacks;
        for (auto& stackConfig : this->config.stacks) {
            stackOrder.push_back(stackConfig.id);
            activeStacks.insert(stackConfig.id);
            stackWindows[stackConfig.id].update(std::move(stackConfig));
        }

        std::vector<std::string> staleStacks;
        for (const auto& [stackId, _] : stackWindows) {
            if (!activeStacks.contains(stackId)) {
                staleStacks.push_back(stackId);
            }
        }
        for (const auto& stackId : staleStacks) {
            stackWindows.erase(stackId);
        }

        // Update detached surface windows.

        detachedSurfaceOrder.clear();
        std::unordered_set<std::string> activeSurfaces;
        for (auto& surfaceConfig : this->config.detachedSurfaces) {
            detachedSurfaceOrder.push_back(surfaceConfig.id);
            activeSurfaces.insert(surfaceConfig.id);
            detachedSurfaceWindows[surfaceConfig.id].update(std::move(surfaceConfig));
        }

        std::vector<std::string> staleSurfaces;
        for (const auto& [surfaceId, _] : detachedSurfaceWindows) {
            if (!activeSurfaces.contains(surfaceId)) {
                staleSurfaces.push_back(surfaceId);
            }
        }
        for (const auto& surfaceId : staleSurfaces) {
            detachedSurfaceWindows.erase(surfaceId);
        }

        if (this->config.empty) {
            hint.update({
                .id = this->config.id + ":empty-hint",
                .title = "Getting Started",
                .subtitle = "Let's get started by adding your first block",
                .steps = {
                    "Double-click anywhere on the canvas.",
                    "Pick a block from the menu.",
                    "Connect blocks to build your flow.",
                },
                .hints = flowgraphHints,
            });
        }
    }

    void render(const Sakura::Context& ctx) {
        window.render(ctx, [this](const Sakura::Context& ctx) {
            editor.render(ctx);
            if (this->config.empty) {
                hint.render(ctx);
            }
            toolbar.render(ctx);
        });
        for (const auto& stackId : stackOrder) {
            if (stackWindows.contains(stackId)) {
                stackWindows.at(stackId).render(ctx);
            }
        }
        for (const auto& surfaceId : detachedSurfaceOrder) {
            if (detachedSurfaceWindows.contains(surfaceId)) {
                detachedSurfaceWindows.at(surfaceId).render(ctx);
            }
        }
    }

 private:
    Config config;
    Sakura::Window window;
    FlowgraphToolbar toolbar;
    FlowgraphEditor editor;
    HintOverlay hint;
    std::vector<std::string> stackOrder;
    std::unordered_map<std::string, FlowgraphStackWindow> stackWindows;
    std::vector<std::string> detachedSurfaceOrder;
    std::unordered_map<std::string, FlowgraphDetachedSurface> detachedSurfaceWindows;
    bool openBlockPickerRequest = false;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_WINDOW_HH
