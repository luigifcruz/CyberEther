#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_CARD_SET_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_CARD_SET_HH

#include <jetstream/render/sakura/components/box.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "helpers.hh"

#include <algorithm>
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
                F32 drop, F32 border = 1.0f, const char* fill = "notification_bg", F32 opacity = 1.0f) {
        const bool visible = opacity > 0.0f;
        const auto color = [&](const char* key) {
            return WithAlpha(ctx.color(key), opacity);
        };
        const F32 line = Px(ctx, 1.0f);
        const F32 outline = Px(ctx, border);
        const ColorRGBA<F32> clear = {0.0f, 0.0f, 0.0f, 0.0f};
        std::vector<Box::Instance> shadowInstances;
        std::vector<Box::Instance> cardInstances;
        std::vector<Box::Instance> outlines;
        std::vector<Box::Instance> rimInstances;
        for (const auto& rect : rects) {
            shadowInstances.push_back({
                .rect = Inflate({rect.x, rect.y + drop, rect.width, rect.height}, softness),
                .visible = visible,
                .backgroundColor = color("tile_shadow"),
            });
            cardInstances.push_back({.rect = rect, .visible = visible, .backgroundColor = color(fill)});
            outlines.push_back({.rect = rect, .visible = visible, .backgroundColor = clear});
            rimInstances.push_back({.rect = Inset(rect, line), .visible = visible, .backgroundColor = clear});
        }
        shadows.update({.instances = std::move(shadowInstances), .clip = clip, .cornerRadius = radius,
                        .softness = softness, .capacity = 4});
        cards.update({.instances = std::move(cardInstances), .clip = clip, .cornerRadius = radius,
                      .borderWidth = outline, .borderColor = color("tile_border"), .strokeFringe = true,
                      .capacity = 4});
        highlights.update({.instances = std::move(outlines), .clip = clip, .cornerRadius = radius, .borderWidth = line,
                           .borderColor = color("tile_highlight"), .strokeFringe = true, .topLit = true,
                           .capacity = 4});
        rims.update({.instances = std::move(rimInstances), .clip = clip,
                     .cornerRadius = std::max(0.0f, radius - line), .borderWidth = line,
                     .borderColor = color("tile_rim"), .strokeFringe = true, .topLit = true, .capacity = 4});
    }
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_CARD_SET_HH
