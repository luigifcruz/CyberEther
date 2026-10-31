#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_LAYOUT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_LAYOUT_HH

#include "labels.hh"

#include "../context.hh"
#include "../tiles.hh"

#include "../../model/tiles.hh"

#include "jetstream/render/sakura/base.hh"

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

inline std::string DescribeTile(const std::string& key) {
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
        return *config + " Config";
    }
    if (auto surface = ParseSurfaceTileKey(key)) {
        return jst::fmt::format("{} ({})", surface->first, surface->second);
    }
    return key;
}

inline TilePaneMap LayoutToDockPanes(const std::string& flowgraphId) {
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

inline TilePaneMap DockToLayoutPanes(const std::string& flowgraphId) {
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

struct LayoutPresenter {
    const PresenterContext& context;

    explicit LayoutPresenter(const PresenterContext& context) : context(context) {}

    std::optional<Sakura::DockTiles> build(const std::string& flowgraphId) const {
        const auto layoutIt = context.state.flowgraph.layouts.find(flowgraphId);
        if (layoutIt == context.state.flowgraph.layouts.end()) {
            return std::nullopt;
        }
        return ToDockTiles(layoutIt->second, LayoutToDockPanes(flowgraphId));
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_LAYOUT_HH
