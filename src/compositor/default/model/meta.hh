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
    std::string windowId;

    JST_SERDES(collapsed, detached, windowId);
};

struct SurfaceMeta {
    U64 attachedHeight = 256;
    U64 detachedWidth = 512;
    U64 detachedHeight = 512;
    bool detached = false;
    bool configCollapsed = true;
    std::string windowId;

    JST_SERDES(attachedHeight, detachedWidth, detachedHeight, detached, configCollapsed, windowId);
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

struct StackDockConfigMeta {
    std::string block;
    U64 order = 0;

    JST_SERDES(block, order);
};

struct StackDockLayoutMeta {
    std::optional<std::string> direction;
    std::optional<F32> ratio;
    std::optional<std::vector<StackDockFlowgraphMeta>> flowgraphs;
    std::optional<std::vector<StackDockSurfaceMeta>> surfaces;
    std::optional<std::vector<StackDockConfigMeta>> configs;
    std::optional<std::vector<StackDockLayoutMeta>> children;

    JST_SERDES(direction, ratio, flowgraphs, surfaces, configs, children);
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
