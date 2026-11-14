#include <jetstream/render/sakura/components/assistant_pill.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kHeight = 28.0f;
constexpr F32 kPadLeft = 10.0f;
constexpr F32 kPadRight = 12.0f;
constexpr F32 kIconGap = 6.0f;
constexpr F32 kIconSize = 13.0f;
constexpr F32 kTextSize = 14.0f;
constexpr F32 kDotGap = 8.0f;
constexpr F32 kDot = 6.0f;
constexpr U64 kShimmerSteps = 32;
constexpr F64 kShimmerPeriod = 1.6;
constexpr F32 kShimmerBandRatio = 0.4f;
constexpr F32 kScanDim = 0.22f;
constexpr F32 kGlowAlpha = 0.22f;
constexpr F32 kGlowInset = 3.0f;

}  // namespace

struct AssistantPill::Impl {
    Config config;
    Box shadow;
    Box base;
    Box fill;
    Box glow;
    Text icon;
    Text label;
    Box shimmer;
    Box highlight;
    Box dot;

    bool hovered = false;
    bool pressed = false;
    bool stalled = false;

    bool working() const {
        return config.state == State::Working;
    }

    void prepare() {
        icon.update({
            .str = config.icon,
            .colorKey = "action_btn_text",
            .scale = kIconSize / Typography::FontSize,
        });
        label.update({
            .str = config.label,
            .font = Text::Font::Display,
            .colorKey = "action_btn_text",
            .scale = kTextSize / Typography::FontSize,
        });
    }
};

AssistantPill::AssistantPill() {
    impl = std::make_unique<Impl>();
    mountChildren({&impl->shadow, &impl->base, &impl->fill, &impl->glow, &impl->icon, &impl->label, &impl->shimmer,
                   &impl->highlight, &impl->dot});
}

AssistantPill::~AssistantPill() = default;

void AssistantPill::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    impl->prepare();
    markConfigured();
}

Extent2D<F32> AssistantPill::measure(const Context& ctx, Extent2D<F32> available) {
    auto& s = *impl;
    const F32 infinite = std::numeric_limits<F32>::infinity();
    F32 width = Px(ctx, kPadLeft) + Px(ctx, kPadRight);
    width += measureChild(s.icon, ctx, {infinite, infinite}).x + Px(ctx, kIconGap);
    width += measureChild(s.label, ctx, {infinite, infinite}).x;
    if (!s.working()) {
        width += Px(ctx, kDotGap) + Px(ctx, kDot);
    }
    return FitExtent({std::ceil(width), Px(ctx, kHeight)}, available);
}

void AssistantPill::layout(const Context& ctx) {
    auto& s = *impl;
    const Rect bounds = frame();
    const bool shown = !bounds.empty();
    const bool working = s.working();
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const F32 radius = bounds.height * 0.5f;

    if (!visible()) {
        s.stalled = true;
    }
    if (shown && working && !s.stalled) {
        requestFrame();
    }

    const F32 softness = Px(ctx, kPillShadowSoftness);
    const F32 drop = Px(ctx, kPillShadowDrop);
    s.shadow.update({
        .instances = {{.rect = {bounds.x - softness, bounds.y + drop - softness, bounds.width + 2.0f * softness,
                                bounds.height + 2.0f * softness},
                       .visible = shown,
                       .backgroundColor = ctx.color("tile_shadow")}},
        .cornerRadius = radius,
        .softness = softness,
        .capacity = 1,
    });
    layoutChild(ctx, s.shadow, bounds);
    s.base.update({
        .instances = {{.rect = bounds, .visible = shown, .backgroundColor = ctx.color(kPillFill)}},
        .cornerRadius = radius,
        .capacity = 1,
    });
    layoutChild(ctx, s.base, bounds);
    const ColorRGBA<F32> surface = ctx.color(s.hovered ? "assistant_btn_hovered" : "assistant_btn");
    s.fill.update({
        .instances = {{.rect = bounds, .visible = shown, .backgroundColor = surface}},
        .cornerRadius = radius,
        .capacity = 1,
    });
    layoutChild(ctx, s.fill, bounds);

    F32 x = bounds.x + Px(ctx, kPadLeft);
    const auto place = [&](Text& text) {
        const Extent2D<F32> size = measureChild(text, ctx, {infinite, infinite});
        const Rect rect = {x, std::round(bounds.y + (bounds.height - size.y) * 0.5f), size.x, size.y};
        layoutChild(ctx, text, rect, bounds);
        x += size.x;
        return rect;
    };
    const Rect iconRect = place(s.icon);
    x += Px(ctx, kIconGap);
    const Rect labelRect = place(s.label);

    const F32 top = std::min(iconRect.y, labelRect.y);
    const F32 bottom = std::max(iconRect.y + iconRect.height, labelRect.y + labelRect.height);
    const Rect zone = {iconRect.x, top, labelRect.x + labelRect.width - iconRect.x, bottom - top};
    std::vector<Box::Instance> strips;
    strips.reserve(kShimmerSteps);
    const F32 step = zone.width / static_cast<F32>(kShimmerSteps);
    const F32 band = std::max(1.0f, zone.width * kShimmerBandRatio);
    const F64 clock = working ? std::max(0.0, ctx.time) : 0.0;
    const F32 phase = static_cast<F32>(std::fmod(clock, kShimmerPeriod) / kShimmerPeriod);
    const F32 center = zone.x - band + phase * (zone.width + 2.0f * band);
    for (U64 k = 0; k < kShimmerSteps; ++k) {
        const F32 position = zone.x + (static_cast<F32>(k) + 0.5f) * step;
        const F32 distance = std::abs(position - center) / band;
        const F32 highlight = distance < 1.0f ? 0.5f * (1.0f + std::cos(std::numbers::pi_v<F32> * distance)) : 0.0f;
        strips.push_back({
            .rect = {zone.x + static_cast<F32>(k) * step, zone.y, step, zone.height},
            .visible = shown && working && !zone.empty(),
            .backgroundColor = WithAlpha(surface, kScanDim * (1.0f - highlight)),
        });
    }
    const F32 inset = Px(ctx, kGlowInset);
    const F32 glowHeight = std::max(0.0f, bounds.height - 2.0f * inset);
    const F32 glowWidth = std::max(glowHeight, band);
    const F32 reach = std::clamp(std::min(center - bounds.x, bounds.x + bounds.width - center) / band, 0.0f, 1.0f);
    s.glow.update({
        .instances = {{
            .rect = {center - glowWidth * 0.5f, bounds.y + inset, glowWidth, glowHeight},
            .visible = shown && working,
            .backgroundColor = WithAlpha(ctx.color("action_btn_text"), kGlowAlpha * Smoothstep(reach)),
        }},
        .clip = bounds,
        .cornerRadius = glowHeight * 0.5f,
        .softness = glowHeight * 0.5f,
        .capacity = 1,
    });
    layoutChild(ctx, s.glow, bounds);
    s.shimmer.update({
        .instances = std::move(strips),
        .clip = bounds,
        .capacity = kShimmerSteps,
    });
    layoutChild(ctx, s.shimmer, bounds);

    s.highlight.update({
        .instances = {{.rect = bounds, .visible = shown, .backgroundColor = {0.0f, 0.0f, 0.0f, 0.0f}}},
        .cornerRadius = radius,
        .borderWidth = Px(ctx, 1.0f),
        .borderColor = ctx.color("action_btn_highlight"),
        .strokeFringe = true,
        .topLit = true,
        .capacity = 1,
    });
    layoutChild(ctx, s.highlight, bounds);

    const F32 dot = Px(ctx, kDot);
    const Rect dotRect = {std::round(x + Px(ctx, kDotGap)), std::round(bounds.y + (bounds.height - dot) * 0.5f), dot,
                          dot};
    s.dot.update({
        .instances = {{.rect = dotRect, .visible = shown && !working, .backgroundColor = ctx.color("action_btn_text")}},
        .cornerRadius = dot * 0.5f,
        .capacity = 1,
    });
    layoutChild(ctx, s.dot, bounds);
}

void AssistantPill::wentOnscreen(U64) {
    if (impl->stalled) {
        impl->stalled = false;
        invalidate(Dirty::Layout);
    }
}

void AssistantPill::wentOffscreen(U64) {
    impl->stalled = true;
}

bool AssistantPill::mouseEvent(const MouseEvent& event) {
    auto& s = *impl;
    const bool inside = frame().contains(event.position.x, event.position.y);
    switch (event.type) {
        case MouseEventType::Move:
            if (s.hovered != inside) {
                s.hovered = inside;
                invalidate(Dirty::Layout);
            }
            return false;
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            if (s.hovered) {
                invalidate(Dirty::Layout);
            }
            s.hovered = false;
            s.pressed = false;
            return false;
        case MouseEventType::Click:
            if (event.button != MouseButton::Left || !inside) {
                return false;
            }
            s.pressed = true;
            return true;
        case MouseEventType::Release: {
            if (event.button != MouseButton::Left || !s.pressed) {
                return false;
            }
            s.pressed = false;
            if (inside && s.config.onClick) {
                s.config.onClick();
            }
            return true;
        }
        default:
            return false;
    }
}

std::optional<SurfaceCursor> AssistantPill::cursorAt(const Extent2D<F32>& point) const {
    if (!frame().contains(point.x, point.y)) {
        return std::nullopt;
    }
    return SurfaceCursor::Hand;
}

}  // namespace Jetstream::Sakura
