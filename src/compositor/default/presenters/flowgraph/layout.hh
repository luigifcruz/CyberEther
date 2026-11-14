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

std::string DescribeTile(const std::string& key);

TilePaneMap LayoutToDockPanes(const std::string& flowgraphId);

TilePaneMap DockToLayoutPanes(const std::string& flowgraphId);

struct LayoutPresenter {
    const PresenterContext& context;

    explicit LayoutPresenter(const PresenterContext& context);
    std::optional<Sakura::DockTiles> build(const std::string& flowgraphId) const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_LAYOUT_HH
