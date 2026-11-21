#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_TILES_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_TILES_HH

#include "../model/tiles.hh"

#include "jetstream/render/sakura/base.hh"

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

using TilePaneMap = std::function<std::optional<std::string>(const std::string&)>;

std::optional<Sakura::DockTiles> ToDockTiles(const TileMeta& meta, const TilePaneMap& map = nullptr);

std::optional<TileMeta> ToTileMeta(const Sakura::DockTiles& tiles, const TilePaneMap& map = nullptr);

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_TILES_HH
