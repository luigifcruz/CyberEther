#include "tiles.hh"

namespace Jetstream {

std::optional<Sakura::DockTiles> ToDockTiles(const TileMeta& meta, const TilePaneMap& map) {
    if (!IsTileLayout(meta)) {
        return std::nullopt;
    }
    Sakura::DockTiles tiles;
    if (!IsTileLeaf(meta)) {
        tiles.vertical = meta.split == "column";
        for (const auto& child : *meta.children) {
            if (auto converted = ToDockTiles(child, map)) {
                tiles.children.push_back(std::move(*converted));
                tiles.sizes.push_back(*child.size);
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
    tiles.minimized = meta.minimized.value_or(false);
    tiles.maximized = meta.maximized.value_or(false);
    return tiles;
}

std::optional<TileMeta> ToTileMeta(const Sakura::DockTiles& tiles, const TilePaneMap& map) {
    TileMeta meta;
    if (!tiles.children.empty()) {
        meta.split = tiles.vertical ? "column" : "row";
        std::vector<TileMeta> children;
        for (U64 index = 0; index < tiles.children.size(); ++index) {
            if (auto child = ToTileMeta(tiles.children[index], map)) {
                child->size = index < tiles.sizes.size() ? tiles.sizes[index] : 0.0f;
                children.push_back(std::move(*child));
            }
        }
        if (children.empty()) {
            return std::nullopt;
        }
        if (children.size() == 1) {
            auto single = std::move(children.front());
            single.size.reset();
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
    if (tiles.minimized) {
        meta.minimized = true;
    }
    if (tiles.maximized) {
        meta.maximized = true;
    }
    return meta;
}

}  // namespace Jetstream
