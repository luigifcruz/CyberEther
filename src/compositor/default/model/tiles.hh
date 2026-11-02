#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_TILES_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_TILES_HH

#include "meta.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

inline constexpr const char* kLayoutEditorPane = "flowgraph";
inline constexpr const char* kLayoutDocumentationPane = "documentation";
inline constexpr const char* kLayoutEnvironmentPane = "environment";
inline constexpr const char* kLayoutDescriptionPane = "description";

inline std::string MakeSurfaceTileKey(const std::string& block, const std::string& surface) {
    return jst::fmt::format("surface:{}:{}", block, surface);
}

inline std::optional<std::pair<std::string, std::string>> ParseSurfaceTileKey(const std::string& key) {
    static const std::string prefix = "surface:";
    if (key.rfind(prefix, 0) != 0) {
        return std::nullopt;
    }

    const std::string rest = key.substr(prefix.size());
    const std::size_t delimiter = rest.find(':');
    if (delimiter == std::string::npos || delimiter == 0 || delimiter + 1 >= rest.size()) {
        return std::nullopt;
    }

    return std::make_pair(rest.substr(0, delimiter), rest.substr(delimiter + 1));
}

inline std::optional<U64> ParseIndexedKey(const std::string& key, const std::string& prefix) {
    if (key.rfind(prefix, 0) != 0 || key.size() == prefix.size()) {
        return std::nullopt;
    }
    U64 index = 0;
    for (const char digit : key.substr(prefix.size())) {
        if (digit < '0' || digit > '9') {
            return std::nullopt;
        }
        index = index * 10 + static_cast<U64>(digit - '0');
    }
    return index;
}

inline std::string MakeTerminalPaneKey(U64 terminal) {
    return jst::fmt::format("terminal:{}", terminal);
}

inline std::optional<U64> ParseTerminalPaneKey(const std::string& key) {
    return ParseIndexedKey(key, "terminal:");
}

inline std::string MakeTerminalTabKey(U64 tab) {
    return jst::fmt::format("terminals:{}", tab);
}

inline std::optional<U64> ParseTerminalTabKey(const std::string& key) {
    return ParseIndexedKey(key, "terminals:");
}

inline std::string MakeConfigTileKey(const std::string& block) {
    return "config:" + block;
}

inline std::optional<std::string> ParseConfigTileKey(const std::string& key) {
    static const std::string prefix = "config:";
    if (key.rfind(prefix, 0) != 0 || key.size() == prefix.size()) {
        return std::nullopt;
    }
    return key.substr(prefix.size());
}

inline std::optional<std::string> TileBlock(const std::string& key) {
    if (auto surface = ParseSurfaceTileKey(key)) {
        return surface->first;
    }
    return ParseConfigTileKey(key);
}

inline bool IsTileLeaf(const TileMeta& tiles) {
    return !tiles.children.has_value() || tiles.children->empty();
}

inline std::optional<TileMeta> MapTiles(const TileMeta& tiles,
                                        const std::function<std::optional<std::string>(const std::string&)>& map) {
    if (IsTileLeaf(tiles)) {
        if (!tiles.pane.has_value() || tiles.pane->empty()) {
            return std::nullopt;
        }
        auto pane = map(*tiles.pane);
        if (!pane.has_value() || pane->empty()) {
            return std::nullopt;
        }
        return TileMeta{.size = tiles.size, .pane = std::move(*pane),
                        .minimized = tiles.minimized, .maximized = tiles.maximized};
    }
    std::vector<TileMeta> children;
    for (const auto& child : *tiles.children) {
        if (auto mapped = MapTiles(child, map)) {
            mapped->size = child.size;
            children.push_back(std::move(*mapped));
        }
    }
    if (children.empty()) {
        return std::nullopt;
    }
    if (children.size() == 1) {
        auto single = std::move(children.front());
        single.size = tiles.size;
        return single;
    }
    return TileMeta{
        .split = tiles.split == "column" ? "column" : "row",
        .size = tiles.size,
        .children = std::move(children),
    };
}

inline bool IsTileLayout(const TileMeta& tiles) {
    if (tiles.split.has_value() && *tiles.split != "row" && *tiles.split != "column") {
        return false;
    }
    if (tiles.size.has_value() && (!std::isfinite(*tiles.size) || *tiles.size < 0.0f)) {
        return false;
    }
    if (IsTileLeaf(tiles)) {
        return tiles.pane.has_value() && !tiles.pane->empty();
    }
    if (!tiles.split.has_value()) {
        return false;
    }
    return std::all_of(tiles.children->begin(), tiles.children->end(), [](const TileMeta& child) {
        return child.size.has_value() && IsTileLayout(child);
    });
}

inline std::optional<TileMeta> FilterTiles(const TileMeta& tiles, const std::function<bool(const std::string&)>& keep) {
    return MapTiles(tiles, [&](const std::string& pane) -> std::optional<std::string> {
        return keep(pane) ? std::optional<std::string>(pane) : std::nullopt;
    });
}

inline void CollectTilePanes(const TileMeta& tiles, std::vector<std::string>& panes) {
    if (IsTileLeaf(tiles)) {
        if (tiles.pane.has_value() && !tiles.pane->empty()) {
            panes.push_back(*tiles.pane);
        }
        return;
    }
    for (const auto& child : *tiles.children) {
        CollectTilePanes(child, panes);
    }
}

inline bool ContainsTilePane(const TileMeta& tiles, const std::string& pane) {
    std::vector<std::string> panes;
    CollectTilePanes(tiles, panes);
    return std::find(panes.begin(), panes.end(), pane) != panes.end();
}

inline bool IsLayoutPane(const std::string& pane) {
    return pane == kLayoutEditorPane || pane == kLayoutEnvironmentPane || pane == kLayoutDescriptionPane ||
           TileBlock(pane).has_value();
}

inline bool IsTransientPane(const std::string& pane) {
    return pane == kLayoutDocumentationPane ||
           ParseTerminalPaneKey(pane).has_value() || ParseTerminalTabKey(pane).has_value();
}

inline TileMeta ReplaceTilePane(const TileMeta& tiles, const std::string& pane, const TileMeta& replacement) {
    if (IsTileLeaf(tiles)) {
        if (tiles.pane != pane) {
            return tiles;
        }
        TileMeta replaced = replacement;
        replaced.size = tiles.size;
        return replaced;
    }
    TileMeta copy = tiles;
    for (auto& child : *copy.children) {
        child = ReplaceTilePane(child, pane, replacement);
    }
    return copy;
}

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_TILES_HH
