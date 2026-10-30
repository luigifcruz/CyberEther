#include <jetstream/render/sakura/components/card.hh>

#include <jetstream/render/sakura/components/box.hh>

#include "../context.hh"
#include "../detail/helpers.hh"
#include "../metrics.hh"

#include <algorithm>
#include <initializer_list>
#include <utility>

namespace Jetstream::Sakura {

struct Card::Impl {
    Config config;
    Box shadow;
    Box fill;
    Box border;
    Box highlight;
    Box rim;
};

Card::Card() {
    impl = std::make_unique<Impl>();
    mountChildren({&impl->shadow, &impl->fill, &impl->border, &impl->highlight, &impl->rim});
}

Card::~Card() = default;

void Card::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
}

void Card::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const bool visible = config.opacity > 0.0f;
    const auto color = [&](const std::string& key) {
        return WithAlpha(ctx.color(key), config.opacity);
    };
    const ColorRGBA<F32> clear = {0.0f, 0.0f, 0.0f, 0.0f};

    std::vector<Box::Instance> shadows;
    std::vector<Box::Instance> fills;
    std::vector<Box::Instance> outlines;
    std::vector<Box::Instance> rims;
    for (const auto& rect : config.rects) {
        shadows.push_back({
            .rect = Inflate({rect.x, rect.y + config.drop, rect.width, rect.height}, config.softness),
            .visible = visible,
            .backgroundColor = color("tile_shadow"),
        });
        fills.push_back({.rect = rect, .visible = visible, .backgroundColor = color(config.fillKey)});
        outlines.push_back({.rect = rect, .visible = visible, .backgroundColor = clear});
        rims.push_back({.rect = Inset(rect, config.lineWidth), .visible = visible, .backgroundColor = clear});
    }

    const bool full = config.layer == Layer::Full;
    const bool base = config.layer != Layer::Frame;
    const bool edge = config.layer != Layer::Base;
    impl->shadow.setVisible(base);
    impl->fill.setVisible(base);
    impl->border.setVisible(!full && edge);
    impl->highlight.setVisible(edge);
    impl->rim.setVisible(edge);

    if (base) {
        impl->shadow.update({
            .instances = std::move(shadows),
            .clip = config.clip,
            .cornerRadius = config.cornerRadius,
            .softness = config.softness,
            .capacity = config.capacity,
        });
        impl->fill.update({
            .instances = std::move(fills),
            .clip = config.clip,
            .cornerRadius = config.cornerRadius,
            .borderWidth = full ? config.borderWidth : 0.0f,
            .borderColor = full ? color("tile_border") : ColorRGBA<F32>{0.0f, 0.0f, 0.0f, 1.0f},
            .strokeFringe = full,
            .capacity = config.capacity,
        });
    }
    if (edge) {
        if (!full) {
            impl->border.update({
                .instances = outlines,
                .clip = config.clip,
                .cornerRadius = config.cornerRadius,
                .borderWidth = config.borderWidth,
                .borderColor = color("tile_border"),
                .strokeFringe = true,
                .capacity = config.capacity,
            });
        }
        impl->highlight.update({
            .instances = std::move(outlines),
            .clip = config.clip,
            .cornerRadius = config.cornerRadius,
            .borderWidth = config.lineWidth,
            .borderColor = color("tile_highlight"),
            .strokeFringe = true,
            .topLit = true,
            .capacity = config.capacity,
        });
        impl->rim.update({
            .instances = std::move(rims),
            .clip = config.clip,
            .cornerRadius = std::max(0.0f, config.cornerRadius - config.lineWidth),
            .borderWidth = config.lineWidth,
            .borderColor = color("tile_rim"),
            .strokeFringe = true,
            .topLit = true,
            .capacity = config.capacity,
        });
    }

    for (auto* part : std::initializer_list<Component*>{&impl->shadow, &impl->fill, &impl->border, &impl->highlight,
                                                         &impl->rim}) {
        if (part->visible()) {
            layoutChild(ctx, *part, bounds);
        }
    }
}

}  // namespace Jetstream::Sakura
