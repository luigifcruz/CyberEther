#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_LAYOUT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_LAYOUT_HH

#include "labels.hh"

#include "../context.hh"

#include "../../model/tiles.hh"

#include "jetstream/render/sakura/base.hh"

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

using TilePaneMap = std::function<std::optional<std::string>(const std::string&)>;

inline std::optional<Sakura::DockTiles> ToDockTiles(const TileMeta& meta, const TilePaneMap& map = nullptr) {
    Sakura::DockTiles tiles;
    if (!IsTileLeaf(meta)) {
        tiles.vertical = meta.split == "column";
        for (const auto& child : *meta.children) {
            if (auto converted = ToDockTiles(child, map)) {
                tiles.children.push_back(std::move(*converted));
                tiles.weights.push_back(child.weight.value_or(1.0f));
            }
        }
        if (tiles.children.empty()) {
            return std::nullopt;
        }
        if (tiles.children.size() == 1) {
            return std::move(tiles.children.front());
        }
        return tiles;
    }
    if (!meta.pane.has_value() || meta.pane->empty()) {
        return std::nullopt;
    }
    auto pane = map ? map(*meta.pane) : meta.pane;
    if (!pane.has_value() || pane->empty()) {
        return std::nullopt;
    }
    tiles.pane = std::move(*pane);
    return tiles;
}

inline std::optional<TileMeta> ToTileMeta(const Sakura::DockTiles& tiles, const TilePaneMap& map = nullptr) {
    TileMeta meta;
    if (!tiles.children.empty()) {
        meta.split = tiles.vertical ? "column" : "row";
        std::vector<TileMeta> children;
        for (U64 index = 0; index < tiles.children.size(); ++index) {
            if (auto child = ToTileMeta(tiles.children[index], map)) {
                const F32 weight = index < tiles.weights.size() ? tiles.weights[index] : 1.0f;
                child->weight = weight > 0.0f ? std::optional<F32>(weight) : std::nullopt;
                children.push_back(std::move(*child));
            }
        }
        if (children.empty()) {
            return std::nullopt;
        }
        if (children.size() == 1) {
            auto single = std::move(children.front());
            single.weight.reset();
            return single;
        }
        meta.children = std::move(children);
        return meta;
    }
    auto pane = map ? map(tiles.pane) : std::optional<std::string>(tiles.pane);
    if (!pane.has_value() || pane->empty()) {
        return std::nullopt;
    }
    meta.pane = std::move(*pane);
    return meta;
}

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
        if (IsTransientPane(pane) || TileBlock(pane).has_value()) {
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
        if (IsTransientPane(pane) || TileBlock(pane).has_value()) {
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
