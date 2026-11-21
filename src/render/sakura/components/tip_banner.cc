#include <jetstream/render/sakura/components/tip_banner.hh>

#include <jetstream/render/sakura/components/card.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/render/tools/icons_lucide.hh>

#include "../context.hh"
#include "../detail/helpers.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr F64 kTipFade = 0.2;
constexpr F64 kTipDelay = 1.0;
constexpr F64 kTipDuration = 8.0;
constexpr F32 kTipHeight = 40.0f;
constexpr F32 kTipPadding = 14.0f;
constexpr const char* kTipTextColor = "text_primary";
constexpr F32 kTipGap = 10.0f;
constexpr F32 kTipText = 14.5f;
constexpr F32 kTipClose = 22.0f;
constexpr F32 kTipCloseIdle = 0.6f;

}  // namespace

struct TipBanner::Impl {
    Config config;
    Card base;
    Text icon;
    Text label;
    Text close;
    Rect closeRect;
    bool started = false;
    bool dismissed = false;
    bool released = false;
    bool pinned = false;
    bool hovered = false;
    bool held = false;
    bool closeHovered = false;
    bool pressed = false;
    F64 start = 0.0;
    F64 seen = 0.0;
    F32 opacity = 0.0f;

    F64 holdStart() const {
        return start + kTipDelay + kTipFade;
    }

    F64 holdEnd() const {
        return holdStart() + kTipDuration;
    }

    F32 fade(TipBanner& self, F64 time) {
        if (dismissed) {
            return 0.0f;
        }
        if (!started) {
            started = true;
            start = time;
            seen = time;
        }
        if (released) {
            released = false;
            if (time >= holdStart()) {
                start = time - kTipDelay - kTipFade;
            }
        }
        if (pinned) {
            pinned = false;
            if (time >= holdEnd() + kTipFade) {
                start = time;
            } else if (time >= holdEnd()) {
                start = time - kTipDelay - kTipFade * static_cast<F64>(opacity);
            }
        }
        if (held && seen >= holdStart() && seen <= holdEnd()) {
            start += time - seen;
        }
        held = hovered;
        seen = time;
        const F64 elapsed = time - start;
        if (elapsed < kTipDelay) {
            self.requestFrame(start + kTipDelay);
            return 0.0f;
        }
        if (elapsed < kTipDelay + kTipFade) {
            self.requestFrame();
            return static_cast<F32>((elapsed - kTipDelay) / kTipFade);
        }
        if (config.persistent) {
            return 1.0f;
        }
        const F64 hold = kTipDelay + kTipFade + kTipDuration;
        if (elapsed < hold) {
            if (!hovered) {
                self.requestFrame(start + hold);
            }
            return 1.0f;
        }
        if (elapsed < hold + kTipFade) {
            self.requestFrame();
            return static_cast<F32>(1.0 - (elapsed - hold) / kTipFade);
        }
        return 0.0f;
    }

    void push() {
        const F32 visible = std::clamp(opacity, 0.0f, 1.0f);
        icon.update({
            .str = config.icon,
            .colorKey = kTipTextColor,
            .opacity = visible,
        });
        label.update({
            .str = config.str,
            .font = Text::Font::Sans,
            .colorKey = kTipTextColor,
            .scale = kTipText / Typography::FontSize,
            .opacity = visible,
        });
        close.update({
            .str = ICON_LC_X,
            .colorKey = kTipTextColor,
            .opacity = visible * (closeHovered ? 1.0f : kTipCloseIdle),
        });
    }
};

TipBanner::TipBanner() {
    impl = std::make_unique<Impl>();
    addChild(impl->base);
    addChild(impl->icon);
    addChild(impl->label);
    addChild(impl->close);
}

TipBanner::~TipBanner() = default;

void TipBanner::update(Config config) {
    auto& s = *impl;
    if (config != s.config) {
        invalidate(Dirty::Layout);
    }
    if (s.config.persistent != config.persistent) {
        s.released = !config.persistent;
        s.pinned = config.persistent;
    }
    s.config = std::move(config);
    s.push();
}

Extent2D<F32> TipBanner::measure(const Context& ctx, Extent2D<F32> available) {
    auto& s = *impl;
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const F32 iconWidth = measureChild(s.icon, ctx, {infinite, infinite}).x;
    const F32 textWidth = measureChild(s.label, ctx, {infinite, infinite}).x;
    const F32 natural = 2.0f * Px(ctx, kTipPadding) + iconWidth + textWidth + Px(ctx, kTipClose) +
                        2.0f * Px(ctx, kTipGap);
    return FitExtent({std::ceil(natural), Px(ctx, kTipHeight)}, available);
}

void TipBanner::layout(const Context& ctx) {
    auto& s = *impl;
    const Rect bounds = frame();
    const F32 opacity = s.fade(*this, ctx.time);
    if (opacity != s.opacity) {
        s.opacity = opacity;
        s.push();
    }
    if (s.opacity <= 0.0f) {
        s.hovered = false;
        s.closeHovered = false;
    }

    const F32 radius = bounds.height * 0.5f;
    const bool drawn = s.opacity > 0.0f && !bounds.empty();
    s.base.update({
        .rects = drawn ? std::vector<Rect>{bounds} : std::vector<Rect>{},
        .cornerRadius = radius,
        .softness = Px(ctx, kPillShadowSoftness),
        .drop = Px(ctx, kPillShadowDrop),
        .borderWidth = Px(ctx, kPillBorder),
        .lineWidth = Px(ctx, 1.0f),
        .fillKey = kPillFill,
        .opacity = s.opacity,
        .capacity = 1,
    });
    layoutChild(ctx, s.base, bounds);

    const F32 infinite = std::numeric_limits<F32>::infinity();
    const F32 padding = Px(ctx, kTipPadding);
    const F32 gap = Px(ctx, kTipGap);
    const F32 centerY = bounds.y + bounds.height * 0.5f;
    const auto place = [&](Text& text, F32 x, F32 limit) {
        const auto size = measureChild(text, ctx, {infinite, infinite});
        const F32 width = std::min(size.x, std::max(0.0f, limit));
        layoutChild(ctx, text, {x, std::round(centerY - size.y * 0.5f), width, size.y}, bounds);
        return width;
    };

    const F32 closeSize = Px(ctx, kTipClose);
    const Rect closeSlot = {bounds.right() - padding - closeSize, std::round(centerY - closeSize * 0.5f), closeSize,
                            closeSize};
    s.closeRect = Intersect(closeSlot, bounds);
    F32 x = bounds.x + padding;
    x += place(s.icon, x, closeSlot.x - x) + gap;
    place(s.label, x, closeSlot.x - gap - x);
    const auto glyph = measureChild(s.close, ctx, {infinite, infinite});
    layoutChild(ctx, s.close,
                {std::round(closeSlot.x + (closeSize - glyph.x) * 0.5f),
                 std::round(closeSlot.y + (closeSize - glyph.y) * 0.5f), glyph.x, glyph.y},
                bounds);
}

bool TipBanner::mouseEvent(const MouseEvent& event) {
    auto& s = *impl;
    const bool live = s.opacity > 0.0f;
    const bool inside = live && frame().contains(event.position.x, event.position.y);
    const bool overClose = inside && s.closeRect.contains(event.position.x, event.position.y);
    switch (event.type) {
        case MouseEventType::Move:
            if (inside != s.hovered || overClose != s.closeHovered) {
                s.hovered = inside;
                s.closeHovered = overClose;
                s.push();
                invalidate(Dirty::Layout);
            }
            return false;
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            if (s.hovered || s.closeHovered || s.pressed) {
                s.hovered = false;
                s.closeHovered = false;
                s.pressed = false;
                s.push();
                invalidate(Dirty::Layout);
            }
            return false;
        case MouseEventType::Click:
            if (!inside) {
                return false;
            }
            if (event.button == MouseButton::Left) {
                s.pressed = overClose;
            }
            return true;
        case MouseEventType::Release:
            if (!s.pressed || event.button != MouseButton::Left) {
                return false;
            }
            s.pressed = false;
            if (overClose) {
                s.dismissed = true;
                s.hovered = false;
                s.closeHovered = false;
                invalidate(Dirty::Layout);
            }
            return true;
        default:
            return false;
    }
}

std::optional<SurfaceCursor> TipBanner::cursorAt(const Extent2D<F32>& point) const {
    if (impl->opacity <= 0.0f || !frame().contains(point.x, point.y) ||
        !impl->closeRect.contains(point.x, point.y)) {
        return std::nullopt;
    }
    return SurfaceCursor::Hand;
}

}  // namespace Jetstream::Sakura
