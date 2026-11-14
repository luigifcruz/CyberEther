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
        Sakura::Callback<void(const std::string&)> onOpenUrl;
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
        std::vector<Sakura::Dock::Pane> terminals;
        std::vector<std::string> adopts;
        std::optional<Description> description;
        bool empty = false;
        bool unsaved = false;
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

    void update(Config config);
    void appendPanes(std::vector<Sakura::Dock::Pane>& panes);
    void appendWindows(std::vector<Sakura::FloatingLayer::Window>& windows);

 private:
    std::pair<const char*, const char*> dependencyGlyph() const;

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
