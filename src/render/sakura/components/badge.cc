#include <jetstream/render/sakura/components/badge.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <cmath>
#include <limits>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kBadgeHeight = 22.0f;
constexpr F32 kBadgePadX = 10.0f;
constexpr F32 kBadgeText = 13.0f;

}  // namespace

struct Badge::Impl {
    Config config;
    Box pill;
    Text label;
};

Badge::Badge() {
    impl = std::make_unique<Impl>();
    mountChildren({&impl->pill, &impl->label});
}

Badge::~Badge() = default;

void Badge::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    impl->label.update({
        .str = impl->config.label,
        .font = Text::Font::Sans,
        .colorKey = impl->config.textColorKey,
        .scale = kBadgeText / Typography::FontSize,
        .weight = kLabelWeight,
    });
}

Extent2D<F32> Badge::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const F32 text = measureChild(impl->label, ctx, {infinite, infinite}).x;
    return FitExtent({std::ceil(text + 2.0f * Px(ctx, kBadgePadX)), Px(ctx, kBadgeHeight)}, available);
}

void Badge::layout(const Context& ctx) {
    auto& s = *impl;
    const Rect bounds = frame();
    const F32 height = std::min(bounds.height, Px(ctx, kBadgeHeight));
    const Rect pill = {bounds.x, bounds.y + std::floor((bounds.height - height) * 0.5f), bounds.width, height};
    s.pill.update({
        .instances = {{.rect = pill, .visible = !pill.empty(), .backgroundColor = ctx.color(s.config.backgroundColorKey)}},
        .cornerRadius = std::min(height * 0.5f, Px(ctx, kFrameRounding)),
        .borderWidth = Px(ctx, 1.0f),
        .borderColor = ctx.color(s.config.borderColorKey),
        .strokeFringe = true,
        .capacity = 1,
    });
    layoutChild(ctx, s.pill, bounds);

    const F32 infinite = std::numeric_limits<F32>::infinity();
    const Extent2D<F32> text = measureChild(s.label, ctx, {infinite, infinite});
    const F32 pad = Px(ctx, kBadgePadX);
    layoutChild(ctx, s.label, {std::round(pill.x + std::max(pad, (pill.width - text.x) * 0.5f)),
                               std::round(pill.y + (pill.height - text.y) * 0.5f), text.x, text.y},
                {pill.x + pad, pill.y, std::max(0.0f, pill.width - 2.0f * pad), pill.height});
}

}  // namespace Jetstream::Sakura
