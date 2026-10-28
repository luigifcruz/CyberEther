#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_PANE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_PANE_HH

#include "../components/hint_overlay.hh"
#include "detached_config.hh"
#include "documentation.hh"
#include "editor/base.hh"
#include "environment.hh"
#include "surface.hh"
#include "../components/callout.hh"

#include "jetstream/render/tools/icons_lucide.hh"

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphPane {
    static constexpr F32 DescriptionBottomPadding = Sakura::Spacing::Small;
    static constexpr F32 DescriptionWidth = 520.0f;
    static constexpr F32 DescriptionFontSize = Sakura::Typography::FontSize * Sakura::Typography::Scale::Lead;

    struct Description {
        std::string id;
        std::string markdown;
        Sakura::Callback<void()> onClose;
    };

    struct Config {
        std::string id;
        std::string title;
        FlowgraphEditor::Config editor;
        std::string layoutGroup;
        std::string editorPane;
        std::optional<Sakura::DockTiles> layout;
        Sakura::Callback<std::string(const std::string&)> describe;
        std::vector<FlowgraphDetachedSurface::Config> detachedSurfaces;
        std::vector<FlowgraphDetachedConfig::Config> detachedConfigs;
        std::optional<FlowgraphDocumentation::Config> documentation;
        std::optional<FlowgraphEnvironment::Config> environment;
        std::optional<Description> description;
        bool empty = false;
        bool showTips = true;
        bool dependencyReviewAvailable = false;
        Callout::Tone dependencyReviewTone = Callout::Tone::Info;
        std::string dependencyReviewMessage;
        Sakura::Callback<void()> onFocus;
        Sakura::Callback<void()> onClose;
        Sakura::Callback<void(std::optional<Sakura::DockTiles>)> onLayout;
        Sakura::Callback<void()> onReviewDependencies;
        Sakura::Callback<void()> onOpenGallery;
        Sakura::Callback<void(FlowgraphEditor::Measured)> onMeasure;
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
            .size = {0.0f, FlowgraphEditor::PillHeight},
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
        if (this->config.environment.has_value()) {
            environment.update(std::move(*this->config.environment));
        }
        if (this->config.description.has_value()) {
            const F32 side = DescriptionFontSize * Sakura::Typography::TextPaddingRatio;
            description.update({
                .id = this->config.description->id + ":markdown",
                .value = this->config.description->markdown,
                .fontSize = DescriptionFontSize,
                .padding = Sakura::Padding{side, 0.0f, side, DescriptionBottomPadding},
                .scrollbar = true,
            });
        }

        if (this->config.empty) {
            hint.update({
                .icon = ICON_LC_SPARKLES,
                .title = "Getting Started",
                .subtitle = "Add your first block, or open a ready-made example.",
                .steps = {
                    "Double-click the canvas or use the add button.",
                    "Pick a block from the list.",
                    "Connect block outputs to inputs.",
                    "Processing starts automatically.",
                },
                .actions = {
                    {
                        .label = ICON_LC_LAYOUT_GRID " Open Gallery",
                        .tone = CardButton::Tone::Primary,
                        .onClick = this->config.onOpenGallery,
                    },
                },
                .width = 380.0f,
            });
        }

        const bool onboarding = this->config.empty && !editor.pickerOpen();
        tip.update({
            .str = flowgraphHints[std::hash<std::string>{}(this->config.id) % flowgraphHints.size()],
            .icon = ICON_LC_LIGHTBULB,
            .persistent = onboarding,
        });

        using Item = Sakura::Overlay::Item;
        std::vector<Item> items = {{.child = &editor.component()}};
        if (onboarding) {
            items.push_back({
                .child = &hint.component(),
                .anchor = Sakura::Anchor::Center,
                .size = hint.size(),
            });
        }
        if (this->config.showTips) {
            items.push_back({
                .child = &tip,
                .anchor = Sakura::Anchor::BottomRight,
                .offset = {-FlowgraphEditor::PillMargin,
                           -(FlowgraphEditor::PillMargin + FlowgraphEditor::PillHeight + FlowgraphEditor::PillGap)},
            });
        }
        if (this->config.dependencyReviewAvailable) {
            items.push_back({
                .child = &dependencyPill,
                .anchor = Sakura::Anchor::TopRight,
                .offset = {-FlowgraphEditor::PillMargin, FlowgraphEditor::PillMargin},
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
        if (this->config.description.has_value()) {
            layoutPanes.push_back({
                .id = this->config.description->id,
                .title = "Description",
                .group = this->config.layoutGroup,
                .icon = ICON_LC_BOOK_OPEN,
                .content = &description,
                .closeOnlyMinimized = true,
                .maximizable = false,
                .leftmost = true,
                .startMinimized = true,
                .width = DescriptionWidth,
                .padding = Sakura::Dock::ContentPadding,
                .onClose = this->config.description->onClose,
            });
        }
        if (this->config.environment.has_value()) {
            layoutPanes.push_back({
                .id = environment.id(),
                .title = environment.title(),
                .group = this->config.layoutGroup,
                .icon = ICON_LC_TABLE_PROPERTIES,
                .content = &environment.component(),
                .padding = Sakura::Dock::ContentPadding,
                .onClose = environment.closeCallback(),
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
            .icon = ICON_LC_WORKFLOW,
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
    Sakura::TipBanner tip;
    std::vector<std::string> detachedSurfaceOrder;
    std::unordered_map<std::string, FlowgraphDetachedSurface> detachedSurfaceViews;
    std::vector<std::string> detachedConfigOrder;
    std::unordered_map<std::string, FlowgraphDetachedConfig> detachedConfigViews;
    FlowgraphDocumentation documentation;
    FlowgraphEnvironment environment;
    Sakura::Markdown description;
    Sakura::Overlay paneOverlay;
    Sakura::Dock layoutDock;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_PANE_HH
