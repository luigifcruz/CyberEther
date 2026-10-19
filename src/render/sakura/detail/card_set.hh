#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_CARD_SET_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_CARD_SET_HH

#include <jetstream/render/sakura/components/box.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "helpers.hh"

#include <vector>

namespace Jetstream::Sakura {

struct CardSet {
    Box shadows;
    Box cards;
    Box highlights;
    Box rims;

    std::vector<Component*> parts() {
        return {&shadows, &cards, &highlights, &rims};
    }

    void update(const Context& ctx, const std::vector<Rect>& rects, const Rect& clip, F32 radius, F32 softness,
                F32 drop, F32 border = 1.0f, const char* fill = "notification_bg", F32 opacity = 1.0f);
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_CARD_SET_HH
