#include <jetstream/render/sakura/components/voice.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/render/tools/icons_fa.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/orb.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kNaturalSize = 175.0f;
constexpr F32 kOrbRadiusRatio = 0.34f;
constexpr F32 kDiagonal = 0.7071f;
constexpr F32 kDismissSize = 22.0f;
constexpr F32 kDismissGlyph = 11.0f;
constexpr F32 kDismissReady = 0.5f;
constexpr F32 kDismissShown = 0.01f;
constexpr F32 kMaxStep = 0.1f;
constexpr F32 kSettled = 0.001f;
constexpr F32 kBusyTime = 0.45f;
constexpr F32 kActivityTime = 0.6f;
constexpr F32 kConnectingActivity = 0.35f;
constexpr F32 kLevelRiseTime = 0.05f;
constexpr F32 kLevelFallTime = 0.18f;
constexpr F32 kHoverTime = 0.08f;

F32 Approach(F32 value, F32 goal, F32 elapsed) {
    value += (goal - value) * (1.0f - std::exp(-elapsed));
    return std::abs(goal - value) < kSettled ? goal : value;
}

}  // namespace

struct Voice::Impl {
    Config config;
    Orb orb;
    Box badge;
    Text glyph;

    F32 busy = 0.0f;
    F32 activity = 0.0f;
    F32 level = 0.0f;
    F32 hover = 0.0f;
    F32 clock = 0.0f;

    bool hovered = false;
    bool dismissHovered = false;
    bool pressedOrb = false;
    bool pressedDismiss = false;
    bool stalled = false;
    Extent2D<F32> center = {0.0f, 0.0f};
    F32 radius = 0.0f;
    Rect dismiss;

    bool startable() const {
        return config.phase == Phase::Idle && static_cast<bool>(config.onStart);
    }

    bool overDismiss(const Extent2D<F32>& point) const {
        return hover > kDismissReady && dismiss.contains(point.x, point.y);
    }

    bool overOrb(const Extent2D<F32>& point) const {
        const F32 dx = point.x - center.x;
        const F32 dy = point.y - center.y;
        return !overDismiss(point) && dx * dx + dy * dy <= radius * radius;
    }
};

Voice::Voice() {
    impl = std::make_unique<Impl>();
    setClipsChildren(true);
    addChild(impl->orb);
    addChild(impl->badge);
    addChild(impl->glyph);
}

Voice::~Voice() = default;

void Voice::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    if (!impl->startable()) {
        impl->pressedOrb = false;
    }
}

Extent2D<F32> Voice::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& config = impl->config;
    const F32 natural = Px(ctx, kNaturalSize);
    return {ResolveLength(ctx, config.size.x, natural, available.x),
            ResolveLength(ctx, config.size.y, natural, available.y)};
}

void Voice::layout(const Context& ctx) {
    auto& state = *impl;
    const auto& config = state.config;
    const Rect bounds = frame();
    const bool shown = !bounds.empty();

    const F32 dt = std::clamp(ctx.deltaTime, 0.0f, kMaxStep);
    const bool active = config.phase != Phase::Idle;
    const bool busy = config.phase == Phase::Busy;
    const bool engaged = busy || config.phase == Phase::Listening;
    const F32 level = active ? std::clamp(config.level, 0.0f, 1.0f) : 0.0f;
    state.busy = Approach(state.busy, busy ? 1.0f : 0.0f, dt / kBusyTime);
    state.activity = Approach(state.activity, engaged ? 1.0f : (active ? kConnectingActivity : 0.0f),
                              dt / kActivityTime);
    state.level = Approach(state.level, level, dt / (level > state.level ? kLevelRiseTime : kLevelFallTime));
    const F32 hoverGoal = state.hovered ? 1.0f : 0.0f;
    state.hover = Approach(state.hover, hoverGoal, dt / kHoverTime);
    const bool animating = active || state.busy != 0.0f || state.activity != 0.0f ||
                           state.level != 0.0f || state.hover != hoverGoal;
    if (animating) {
        state.clock += dt;
    }
    if (!visible()) {
        state.stalled = true;
    }
    if (shown && !state.stalled && animating) {
        requestFrame();
    }

    state.radius = std::max(0.0f, std::min(bounds.width, bounds.height) * kOrbRadiusRatio);
    state.center = {bounds.x + bounds.width * 0.5f, bounds.y + bounds.height * 0.5f};
    state.orb.update({
        .rect = bounds,
        .center = state.center,
        .radius = state.radius,
        .time = state.clock,
        .busy = state.busy,
        .activity = state.activity,
        .level = state.level,
        .listeningColor = ctx.color("voice_orb_listening"),
        .busyColor = ctx.color("voice_orb_busy"),
        .listeningAccent = ctx.color("voice_orb_listening_accent"),
        .busyAccent = ctx.color("voice_orb_busy_accent"),
        .clip = bounds,
    });
    layoutChild(ctx, state.orb, bounds);

    const F32 badge = Px(ctx, kDismissSize);
    const F32 diagonal = state.radius * kDiagonal;
    state.dismiss = {std::round(state.center.x + diagonal - badge * 0.5f),
                     std::round(state.center.y - diagonal - badge * 0.5f), badge, badge};
    const F32 alpha = std::clamp(state.hover, 0.0f, 1.0f);
    const bool badgeShown = shown && alpha > kDismissShown;
    const char* fill = state.pressedDismiss && state.dismissHovered ? "button_active"
                       : state.dismissHovered                       ? "button_hovered"
                                                                    : "card";
    state.badge.update({
        .instances = {{
            .rect = state.dismiss,
            .visible = badgeShown,
            .backgroundColor = WithAlpha(ctx.color(fill), alpha),
        }},
        .clip = bounds,
        .cornerRadius = badge * 0.5f,
        .borderWidth = Px(ctx, 1.0f),
        .borderColor = WithAlpha(ctx.color("border"), alpha),
    });
    layoutChild(ctx, state.badge, bounds);

    state.glyph.update({
        .str = ICON_FA_XMARK,
        .colorKey = "text_secondary",
        .scale = kDismissGlyph / Typography::FontSize,
        .opacity = badgeShown ? alpha : 0.0f,
    });
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const Extent2D<F32> glyph = measureChild(state.glyph, ctx, {infinite, infinite});
    layoutChild(ctx, state.glyph,
                {std::round(state.dismiss.x + (badge - glyph.x) * 0.5f),
                 std::round(state.dismiss.y + (badge - glyph.y) * 0.5f), glyph.x, glyph.y},
                state.dismiss);
}

void Voice::wentOnscreen(U64) {
    if (impl->stalled) {
        impl->stalled = false;
        invalidate(Dirty::Layout);
    }
}

void Voice::wentOffscreen(U64) {
    impl->stalled = true;
}

bool Voice::mouseEvent(const MouseEvent& event) {
    auto& state = *impl;
    const Extent2D<F32> point = event.position;
    const bool inside = frame().contains(point.x, point.y);
    const bool overDismiss = inside && state.overDismiss(point);
    switch (event.type) {
        case MouseEventType::Move:
            if (state.hovered != inside || state.dismissHovered != overDismiss) {
                state.hovered = inside;
                state.dismissHovered = overDismiss;
                invalidate(Dirty::Layout);
            }
            return false;
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            if (state.hovered || state.dismissHovered || state.pressedDismiss) {
                invalidate(Dirty::Layout);
            }
            state.hovered = false;
            state.dismissHovered = false;
            state.pressedDismiss = false;
            state.pressedOrb = false;
            return false;
        case MouseEventType::Click:
            if (event.button != MouseButton::Left || !inside) {
                return false;
            }
            if (overDismiss) {
                state.pressedDismiss = true;
                state.dismissHovered = true;
                invalidate(Dirty::Layout);
                return true;
            }
            state.pressedOrb = state.startable() && state.overOrb(point);
            return state.pressedOrb;
        case MouseEventType::Release: {
            if (event.button != MouseButton::Left) {
                return false;
            }
            if (state.pressedDismiss) {
                state.pressedDismiss = false;
                invalidate(Dirty::Layout);
                if (overDismiss && state.config.onDismiss) {
                    state.config.onDismiss();
                }
                return true;
            }
            const bool start = state.pressedOrb && inside && state.startable() && state.overOrb(point);
            state.pressedOrb = false;
            if (start) {
                state.config.onStart();
                return true;
            }
            return false;
        }
        default:
            return false;
    }
}

std::optional<SurfaceCursor> Voice::cursorAt(const Extent2D<F32>& point) const {
    if (!frame().contains(point.x, point.y)) {
        return std::nullopt;
    }
    if (impl->overDismiss(point) || (impl->startable() && impl->overOrb(point))) {
        return SurfaceCursor::Hand;
    }
    return std::nullopt;
}

}  // namespace Jetstream::Sakura
