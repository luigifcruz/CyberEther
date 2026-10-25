#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_PANE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_PANE_HH

#include "../components/hint_overlay.hh"
#include "detached_config.hh"
#include "documentation.hh"
#include "editor/base.hh"
#include "surface.hh"
#include "../components/callout.hh"

#include "jetstream/render/tools/icons_lucide.hh"

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphPane {
    struct Config {
        std::string id;
        std::string title;
        FlowgraphEditor::Config editor;
        std::string layoutGroup;
        std::string editorPane;
        std::optional<Sakura::DockTiles> layout;
        std::function<std::string(const std::string&)> describe;
        std::vector<FlowgraphDetachedSurface::Config> detachedSurfaces;
        std::vector<FlowgraphDetachedConfig::Config> detachedConfigs;
        std::optional<FlowgraphDocumentation::Config> documentation;
        bool empty = false;
        bool dependencyReviewAvailable = false;
        Callout::Tone dependencyReviewTone = Callout::Tone::Info;
        std::string dependencyReviewMessage;
        std::function<void()> onFocus;
        std::function<void()> onClose;
        std::function<void(std::optional<Sakura::DockTiles>)> onLayout;
        std::function<void()> onReviewDependencies;
        std::function<void()> onOpenExamples;
        std::function<void(FlowgraphEditor::Measured)> onMeasure;
    };

    void update(Config config) {
        auto editorConfig = std::move(config.editor);
        editorConfig.onLayout = std::move(config.onMeasure);
        this->config = std::move(config);

        const std::vector<std::string> flowgraphHints = {
            "Drag from output to input to connect blocks.",
            "Drag blocks to move them around.",
            "Right-click a block to open its context menu.",
            "Use Ctrl+C and Ctrl+V to copy and paste blocks.",
            "Hover over ports to see their data type.",
            "Source blocks set the buffer size downstream.",
            "Click and drag on empty canvas to pan around.",
            "Hover over diagnostic icons for details.",
            "Right-click and reload a block to retry.",
            "Right-click a block for its documentation.",
        };

        const auto [dependencyIcon, dependencyColor] = dependencyGlyph();
        dependencyPill.update({
            .icon = dependencyIcon,
            .iconColorKey = dependencyColor,
            .placeholder = this->config.dependencyReviewMessage,
            .size = {0.0f, FlowgraphBlockPicker::PillHeight},
            .onOpen = this->config.onReviewDependencies,
        });
        editor.update(std::move(editorConfig));

        // Update detached surface windows.

        detachedSurfaceOrder.clear();
        std::unordered_set<std::string> activeSurfaces;
        for (auto& surfaceConfig : this->config.detachedSurfaces) {
            detachedSurfaceOrder.push_back(surfaceConfig.id);
            activeSurfaces.insert(surfaceConfig.id);
            detachedSurfaceViews[surfaceConfig.id].update(std::move(surfaceConfig));
        }

        std::vector<std::string> staleSurfaces;
        for (const auto& [surfaceId, _] : detachedSurfaceViews) {
            if (!activeSurfaces.contains(surfaceId)) {
                staleSurfaces.push_back(surfaceId);
            }
        }
        for (const auto& surfaceId : staleSurfaces) {
            detachedSurfaceViews.erase(surfaceId);
        }

        // Update detached config windows.

        detachedConfigOrder.clear();
        std::unordered_set<std::string> activeConfigs;
        for (auto& detachedConfig : this->config.detachedConfigs) {
            detachedConfigOrder.push_back(detachedConfig.id);
            activeConfigs.insert(detachedConfig.id);
            detachedConfigViews[detachedConfig.id].update(std::move(detachedConfig));
        }

        std::vector<std::string> staleConfigs;
        for (const auto& [configId, _] : detachedConfigViews) {
            if (!activeConfigs.contains(configId)) {
                staleConfigs.push_back(configId);
            }
        }
        for (const auto& configId : staleConfigs) {
            detachedConfigViews.erase(configId);
        }

        if (this->config.documentation.has_value()) {
            documentation.update(std::move(*this->config.documentation));
        }

        if (this->config.empty) {
            hint.update({
                .id = this->config.id + ":empty-hint",
                .title = "Getting Started",
                .subtitle = "Add your first block, or open a ready-made example.",
                .steps = {
                    "Double-click the canvas or use Search blocks.",
                    "Pick a block from the list.",
                    "Connect block outputs to inputs.",
                    "Processing starts automatically.",
                },
                .hints = flowgraphHints,
                .actions = {
                    {
                        .id = "open-examples",
                        .label = ICON_LC_LAYOUT_GRID " Open Examples",
                        .variant = Sakura::Button::Variant::Action,
                        .onClick = this->config.onOpenExamples,
                    },
                },
                .size = {420.0f, 430.0f},
            });
        }

        using Item = Sakura::Overlay::Item;
        std::vector<Item> items = {{.child = &editor.component()}};
        if (this->config.empty) {
            items.push_back({
                .child = &hint.component(),
                .anchor = Sakura::Anchor::Center,
                .size = hint.size(),
            });
        }
        if (this->config.dependencyReviewAvailable) {
            items.push_back({
                .child = &dependencyPill,
                .anchor = Sakura::Anchor::TopRight,
                .offset = {-(FlowgraphBlockPicker::PillMargin + FlowgraphBlockPicker::PillWidth + FlowgraphEditor::PillGap +
                             FlowgraphEditor::ZoomPillWidth + FlowgraphEditor::PillGap),
                           FlowgraphBlockPicker::PillMargin},
            });
        }
        paneOverlay.update({.items = std::move(items)});

        std::vector<Sakura::Dock::Pane> layoutPanes = {{
            .id = this->config.editorPane,
            .title = "Flowgraph",
            .group = this->config.layoutGroup,
            .icon = ICON_LC_BOXES,
            .content = &paneOverlay,
            .closable = false,
            .padding = Sakura::Dock::ContentPadding,
        }};
        for (const auto& surfaceId : detachedSurfaceOrder) {
            auto& surface = detachedSurfaceViews.at(surfaceId);
            layoutPanes.push_back({
                .id = surfaceId,
                .title = surface.title(),
                .group = this->config.layoutGroup,
                .icon = ICON_LC_CHART_AREA,
                .content = &surface.component(),
                .padding = Sakura::Dock::ContentPadding,
                .onClose = surface.closeCallback(),
                .actions = surface.actions(),
            });
        }
        for (const auto& configId : detachedConfigOrder) {
            auto& detached = detachedConfigViews.at(configId);
            layoutPanes.push_back({
                .id = configId,
                .title = detached.title(),
                .group = this->config.layoutGroup,
                .icon = ICON_LC_SLIDERS_HORIZONTAL,
                .content = &detached.component(),
                .padding = Sakura::Dock::ContentPadding,
                .onClose = detached.closeCallback(),
            });
        }
        if (this->config.documentation.has_value()) {
            layoutPanes.push_back({
                .id = documentation.id(),
                .title = documentation.title(),
                .group = this->config.layoutGroup,
                .icon = ICON_LC_BOOK,
                .content = &documentation.component(),
                .padding = Sakura::Dock::ContentPadding,
                .onClose = documentation.closeCallback(),
            });
        }
        layoutDock.update({
            .tabs = false,
            .accepts = this->config.layoutGroup,
            .panes = std::move(layoutPanes),
            .tiles = this->config.layout,
            .describe = this->config.describe,
            .onTiles = this->config.onLayout,
            .columns = true,
            .flexPane = this->config.editorPane,
        });
    }

    void appendPanes(std::vector<Sakura::Dock::Pane>& panes) {
        panes.push_back({
            .id = config.id,
            .title = config.title.empty() ? config.id : config.title,
            .group = "main",
            .content = &layoutDock,
            .onClose = config.onClose,
            .onFocus = config.onFocus,
        });
    }

    void appendWindows(std::vector<Sakura::FloatingLayer::Window>& windows) {
        editor.appendWindows(windows);
    }

 private:
    std::pair<const char*, const char*> dependencyGlyph() const {
        switch (config.dependencyReviewTone) {
            case Callout::Tone::Error:
                return {ICON_LC_CIRCLE_ALERT, "error_red"};
            case Callout::Tone::Warning:
                return {ICON_LC_TRIANGLE_ALERT, "warning_yellow"};
            case Callout::Tone::Success:
                return {ICON_LC_CIRCLE_CHECK, "success_green"};
            case Callout::Tone::Info:
                break;
        }
        return {ICON_LC_PACKAGE, "info_blue"};
    }

    Config config;
    Sakura::PillField dependencyPill;
    FlowgraphEditor editor;
    HintOverlay hint;
    std::vector<std::string> detachedSurfaceOrder;
    std::unordered_map<std::string, FlowgraphDetachedSurface> detachedSurfaceViews;
    std::vector<std::string> detachedConfigOrder;
    std::unordered_map<std::string, FlowgraphDetachedConfig> detachedConfigViews;
    FlowgraphDocumentation documentation;
    Sakura::Overlay paneOverlay;
    Sakura::Dock layoutDock;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_PANE_HH
