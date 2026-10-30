#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_CLIP_LAYER_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_CLIP_LAYER_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <vector>

namespace Jetstream::Sakura {

struct ClipLayer : public Component {
    ClipLayer() {
        setClipsChildren(true);
    }

    void mount(const std::vector<Component*>& children) {
        mountChildren(children);
    }

    void place(const Context& ctx, Component& child, Rect rect) {
        layoutChild(ctx, child, rect);
    }

 protected:
    void layout(const Context&) override {}
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_CLIP_LAYER_HH
