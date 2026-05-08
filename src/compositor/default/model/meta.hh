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

struct SurfaceMeta {
    U64 attachedWidth = 256;
    U64 attachedHeight = 256;
    U64 detachedWidth = 512;
    U64 detachedHeight = 512;
    bool detached = false;

    JST_SERDES(attachedWidth, attachedHeight, detachedWidth, detachedHeight, detached);
};

struct StackDockFlowgraphMeta {
    U64 order = 0;

    JST_SERDES(order);
};

struct StackDockSurfaceMeta {
    std::string block;
    std::string surface;
    U64 order = 0;

    JST_SERDES(block, surface, order);
};

struct StackDockLayoutMeta {
    std::optional<std::string> direction;
    std::optional<F32> ratio;
    std::optional<std::vector<StackDockFlowgraphMeta>> flowgraphs;
    std::optional<std::vector<StackDockSurfaceMeta>> surfaces;
    std::optional<std::vector<StackDockLayoutMeta>> children;

    JST_SERDES(direction, ratio, flowgraphs, surfaces, children);
};

struct StackMeta {
    std::string title;
    F32 x = 0.0f;
    F32 y = 0.0f;
    F32 width = 500.0f;
    F32 height = 300.0f;
    std::optional<StackDockLayoutMeta> layout;

    JST_SERDES(title, x, y, width, height, layout);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_META_HH
