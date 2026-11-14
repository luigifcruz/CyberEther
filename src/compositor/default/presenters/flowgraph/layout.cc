#include "layout.hh"

namespace Jetstream {

std::string DescribeTile(const std::string& key) {
    if (key == kLayoutDocumentationPane) {
        return "Documentation";
    }
    if (key == kLayoutEnvironmentPane) {
        return "Environment";
    }
    if (key == kLayoutDescriptionPane) {
        return "Description";
    }
    if (ParseTerminalPaneKey(key).has_value()) {
        return "Terminal";
    }
    if (auto config = ParseConfigTileKey(key)) {
        return *config + " Controls";
    }
    if (auto surface = ParseSurfaceTileKey(key)) {
        return jst::fmt::format("{} ({})", surface->first, surface->second);
    }
    return key;
}

TilePaneMap LayoutToDockPanes(const std::string& flowgraphId) {
    return [flowgraphId](const std::string& pane) -> std::optional<std::string> {
        if (pane == kLayoutEditorPane) {
            return MakeEditorPaneId(flowgraphId);
        }
        if (IsTransientPane(pane) || IsLayoutPane(pane)) {
            return pane;
        }
        return std::nullopt;
    };
}

TilePaneMap DockToLayoutPanes(const std::string& flowgraphId) {
    return [flowgraphId](const std::string& pane) -> std::optional<std::string> {
        if (pane == MakeEditorPaneId(flowgraphId)) {
            return std::string(kLayoutEditorPane);
        }
        if (IsTransientPane(pane) || IsLayoutPane(pane)) {
            return pane;
        }
        return std::nullopt;
    };
}

LayoutPresenter::LayoutPresenter(const PresenterContext& context) : context(context) {}

std::optional<Sakura::DockTiles> LayoutPresenter::build(const std::string& flowgraphId) const {
    const auto layoutIt = context.state.flowgraph.layouts.find(flowgraphId);
    if (layoutIt == context.state.flowgraph.layouts.end()) {
        return std::nullopt;
    }
    return ToDockTiles(layoutIt->second, LayoutToDockPanes(flowgraphId));
}

}  // namespace Jetstream
