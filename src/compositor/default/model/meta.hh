#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_META_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_META_HH

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

struct ConfigMeta {
    bool collapsed = false;
    bool detached = false;

    JST_SERDES(collapsed, detached);
};

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

    JST_SERDES(split, weight, pane, children);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_META_HH
