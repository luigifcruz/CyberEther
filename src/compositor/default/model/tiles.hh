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

std::string MakeSurfaceTileKey(const std::string& block, const std::string& surface);

std::optional<std::pair<std::string, std::string>> ParseSurfaceTileKey(const std::string& key);

std::string MakeTerminalPaneKey(U64 terminal);

std::optional<U64> ParseTerminalPaneKey(const std::string& key);

std::string MakeTerminalTabKey(U64 tab);

std::optional<U64> ParseTerminalTabKey(const std::string& key);

std::string MakeConfigTileKey(const std::string& block);

std::optional<std::string> ParseConfigTileKey(const std::string& key);

std::optional<std::string> TileBlock(const std::string& key);

bool IsTileLeaf(const TileMeta& tiles);

std::optional<TileMeta> MapTiles(const TileMeta& tiles,
                                 const std::function<std::optional<std::string>(const std::string&)>& map);

bool IsTileLayout(const TileMeta& tiles);

std::optional<TileMeta> FilterTiles(const TileMeta& tiles, const std::function<bool(const std::string&)>& keep);

void CollectTilePanes(const TileMeta& tiles, std::vector<std::string>& panes);

bool ContainsTilePane(const TileMeta& tiles, const std::string& pane);

bool IsLayoutPane(const std::string& pane);

bool IsTransientPane(const std::string& pane);

TileMeta ReplaceTilePane(const TileMeta& tiles, const std::string& pane, const TileMeta& replacement);

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_TILES_HH
