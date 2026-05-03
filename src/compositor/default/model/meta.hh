#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_META_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_META_HH

#include "jetstream/parser.hh"

namespace Jetstream {

struct NodeMeta {
    F32 x = 0.0f;
    F32 y = 0.0f;
    F32 width = 0.0f;
    F32 height = 0.0f;

    JST_SERDES(x, y, width, height);
};

struct SurfaceMeta {
    U64 attachedWidth = 512;
    U64 attachedHeight = 512;
    U64 detachedWidth = 512;
    U64 detachedHeight = 512;

    JST_SERDES(attachedWidth, attachedHeight, detachedWidth, detachedHeight);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_META_HH
