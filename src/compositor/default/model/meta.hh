#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_META_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_META_HH

#include "jetstream/block.hh"
#include "jetstream/parser.hh"

#include <optional>
#include <string>
#include <vector>

namespace Jetstream {

struct NodeMeta {
    F32 x = 0.0f;
    F32 y = 0.0f;
    F32 width = 0.0f;
    F32 height = 0.0f;

    JST_SERDES(x, y, width, height);
};

struct EditorMeta {
    F32 x = 0.0f;
    F32 y = 0.0f;
    F32 zoom = 1.0f;

    JST_SERDES(x, y, zoom);
};

struct ConfigMeta {
    bool collapsed = false;
    bool detached = false;
    bool tileMetricsCollapsed = false;

    JST_SERDES(collapsed, detached, tileMetricsCollapsed);
};

constexpr F32 DefaultNodeWidth(Block::NodeSize size) {
    switch (size) {
        case Block::NodeSize::XS:
            return 120.0f;
        case Block::NodeSize::M:
            return 220.0f;
        case Block::NodeSize::L:
            return 320.0f;
        case Block::NodeSize::XL:
            return 460.0f;
        default:
            return 140.0f;
    }
}

struct SurfaceMeta {
    U64 attachedHeight = 256;
    bool detached = false;
    bool configCollapsed = true;

    JST_SERDES(attachedHeight, detached, configCollapsed);
};

struct TileMeta {
    std::optional<std::string> split;
    std::optional<F32> weight;
    std::optional<std::string> pane;
    std::optional<std::vector<TileMeta>> children;
    std::optional<bool> minimized;
    std::optional<bool> maximized;

    JST_SERDES(split, weight, pane, children, minimized, maximized);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_META_HH
