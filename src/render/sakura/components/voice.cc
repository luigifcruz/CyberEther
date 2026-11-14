#include <jetstream/render/sakura/components/voice.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/card.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/render/tools/icons_lucide.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/voice_wave.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kWidth = 140.0f;
constexpr F32 kHeight = 28.0f;
constexpr F32 kWaveMargin = 1.5f;
constexpr F32 kPitch = 2.5f;
constexpr F32 kDot = 0.85f;
constexpr F32 kHintText = 13.0f;
constexpr F32 kHintWeight = 0.2f;
constexpr F32 kCoverAlpha = 0.82f;
constexpr F32 kOutline = 1.5f;
constexpr F32 kMaxStep = 0.1f;
constexpr F32 kSettled = 0.001f;
constexpr F32 kBusyTime = 0.45f;
constexpr F32 kActivityTime = 0.6f;
constexpr F32 kConnectingActivity = 0.35f;
constexpr F32 kLevelRiseTime = 0.05f;
constexpr F32 kLevelFallTime = 0.18f;
constexpr F32 kHoverTime = 0.08f;
constexpr F32 kMuteGap = 6.0f;
constexpr F32 kMuteIcon = 14.0f;

F32 Approach(F32 value, F32 goal, F32 elapsed) {
    value += (goal - value) * (1.0f - std::exp(-elapsed));
    return std::abs(goal - value) < kSettled ? goal : value;
}

}  // namespace

struct Voice::Impl {
    Config config;
    Card base;
    VoiceWave wave;
    Box cover;
    Text hint;
    Box muteCover;
    Text muteIcon;

    Rect pill;
    Rect mute;

    F32 busy = 0.0f;
    F32 activity = 0.0f;
    F32 level = 0.0f;
    F32 hover = 0.0f;
    F32 muteHover = 0.0f;
    F32 clock = 0.0f;

    bool hovered = false;
    bool hoveredMute = false;
    bool pressed = false;
    bool pressedMute = false;
    bool starting = false;
    bool stalled = false;

    bool mutes() const {
        return static_cast<bool>(config.onMute);
    }

    bool restarts() const {
        return config.phase == Phase::Idle && static_cast<bool>(config.onStart);
    }

    void split(const Context& ctx, const Rect& bounds) {
        if (!mutes()) {
            mute = {};
            pill = bounds;
            hoveredMute = false;
            pressedMute = false;
            return;
        }
        const F32 side = std::min(bounds.width, bounds.height);
        const F32 offset = side + Px(ctx, kMuteGap);
        mute = {bounds.x, bounds.y, side, bounds.height};
        pill = {bounds.x + offset, bounds.y, std::max(0.0f, bounds.width - offset), bounds.height};
    }

    void layoutMute(Voice& self, const Context& ctx, bool shown) {
        const F32 hover = std::clamp(muteHover, 0.0f, 1.0f);
        const char* accent = config.muted ? "voice_stop" : "text_primary";
        muteCover.update({
            .instances = {{
                .rect = mute,
                .visible = shown && !mute.empty() && hover > 0.0f,
                .backgroundColor = ctx.color("transparent"),
            }},
            .clip = mute,
            .cornerRadius = mute.height * 0.5f,
            .borderWidth = Px(ctx, kOutline),
            .borderColor = WithAlpha(ctx.color(accent), hover),
            .strokeFringe = true,
            .capacity = 1,
        });
        self.layoutChild(ctx, muteCover, mute);

        muteIcon.setVisible(!mute.empty());
        muteIcon.update({
            .str = config.muted ? ICON_LC_MIC_OFF : ICON_LC_MIC,
            .align = Text::Align::Center,
            .colorKey = config.muted ? "voice_stop" : (hoveredMute ? "text_primary" : "voice_dim"),
            .scale = kMuteIcon / Typography::FontSize,
        });
        const F32 height = self.measureChild(muteIcon, ctx, {mute.width, std::numeric_limits<F32>::infinity()}).y;
        self.layoutChild(ctx, muteIcon, {mute.x, std::round(mute.y + (mute.height - height) * 0.5f), mute.width, height},
                         mute);
    }
};

Voice::Voice() {
    impl = std::make_unique<Impl>();
    mountChildren({&impl->base, &impl->wave, &impl->cover, &impl->hint, &impl->muteCover, &impl->muteIcon});
}

Voice::~Voice() = default;

void Voice::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    markConfigured();
}

Extent2D<F32> Voice::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 mute = impl->mutes() ? kHeight + kMuteGap : 0.0f;
    return FitExtent({Px(ctx, kWidth + mute), Px(ctx, kHeight)}, available);
}

void Voice::layout(const Context& ctx) {
    auto& s = *impl;
    const auto& config = s.config;
    const bool shown = !frame().empty();
    s.split(ctx, frame());
    const Rect bounds = s.pill;

    const F32 dt = std::clamp(ctx.deltaTime, 0.0f, kMaxStep);
    const bool active = config.phase != Phase::Idle;
    const bool busy = config.phase == Phase::Busy;
    const bool engaged = busy || config.phase == Phase::Listening;
    const F32 level = active ? std::clamp(config.level, 0.0f, 1.0f) : 0.0f;
    const F32 hoverGoal = s.hovered ? 1.0f : 0.0f;
    s.busy = Approach(s.busy, busy ? 1.0f : 0.0f, dt / kBusyTime);
    s.activity = Approach(s.activity, engaged ? 1.0f : (active ? kConnectingActivity : 0.0f), dt / kActivityTime);
    s.level = Approach(s.level, level, dt / (level > s.level ? kLevelRiseTime : kLevelFallTime));
    s.hover = Approach(s.hover, hoverGoal, dt / kHoverTime);
    const F32 muteHoverGoal = s.hoveredMute ? 1.0f : 0.0f;
    s.muteHover = Approach(s.muteHover, muteHoverGoal, dt / kHoverTime);
    const bool animating = active || s.busy != 0.0f || s.activity != 0.0f || s.level != 0.0f ||
                           s.hover != hoverGoal || s.muteHover != muteHoverGoal;
    if (animating) {
        s.clock += dt;
    }
    if (!visible()) {
        s.stalled = true;
    }
    if (shown && !s.stalled && animating) {
        requestFrame();
    }

    const F32 radius = bounds.height * 0.5f;
    std::vector<Rect> cards;
    for (const Rect& rect : {s.mute, bounds}) {
        if (shown && !rect.empty()) {
            cards.push_back(rect);
        }
    }
    s.base.update({
        .rects = std::move(cards),
        .cornerRadius = radius,
        .softness = Px(ctx, kPillShadowSoftness),
        .drop = Px(ctx, kPillShadowDrop),
        .borderWidth = Px(ctx, 1.0f),
        .lineWidth = Px(ctx, 1.0f),
        .fillKey = "voice_bg",
        .capacity = 2,
    });
    layoutChild(ctx, s.base, frame());

    s.wave.update({
        .rect = bounds,
        .pitch = Px(ctx, kPitch),
        .dot = Px(ctx, kDot),
        .cornerRadius = radius,
        .margin = Px(ctx, kDot + kWaveMargin),
        .time = s.clock,
        .busy = s.busy,
        .activity = s.activity,
        .level = s.level,
        .listeningColor = ctx.color("voice_listening"),
        .busyColor = ctx.color("voice_busy"),
        .dimColor = ctx.color("voice_dim"),
        .clip = bounds,
    });
    layoutChild(ctx, s.wave, bounds);

    const F32 hover = std::clamp(s.hover, 0.0f, 1.0f);
    const char* accent = s.restarts() ? "text_primary" : "voice_stop";
    s.cover.update({
        .instances = {{
            .rect = bounds,
            .visible = shown && hover > 0.0f,
            .backgroundColor = WithAlpha(ctx.color("voice_bg"), kCoverAlpha * hover),
        }},
        .clip = bounds,
        .cornerRadius = radius,
        .borderWidth = Px(ctx, kOutline),
        .borderColor = WithAlpha(ctx.color(accent), hover),
        .strokeFringe = true,
        .capacity = 1,
    });
    layoutChild(ctx, s.cover, bounds);

    s.hint.update({
        .str = s.restarts() ? config.startHint : config.stopHint,
        .font = Text::Font::Sans,
        .align = Text::Align::Center,
        .colorKey = accent,
        .scale = kHintText / Typography::FontSize,
        .weight = kHintWeight,
        .opacity = hover,
    });
    const F32 height = measureChild(s.hint, ctx, {bounds.width, std::numeric_limits<F32>::infinity()}).y;
    layoutChild(ctx, s.hint, {bounds.x, std::round(bounds.y + (bounds.height - height) * 0.5f), bounds.width, height},
                bounds);

    s.layoutMute(*this, ctx, shown);
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
    auto& s = *impl;
    const Extent2D<F32> point = event.position;
    const bool inside = s.pill.contains(point.x, point.y);
    const bool insideMute = !s.mute.empty() && s.mute.contains(point.x, point.y);
    switch (event.type) {
        case MouseEventType::Move:
            if (s.hovered != inside || s.hoveredMute != insideMute) {
                s.hovered = inside;
                s.hoveredMute = insideMute;
                invalidate(Dirty::Layout);
            }
            return false;
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            if (s.hovered || s.hoveredMute) {
                invalidate(Dirty::Layout);
            }
            s.hovered = false;
            s.hoveredMute = false;
            s.pressed = false;
            s.pressedMute = false;
            return false;
        case MouseEventType::Click:
            if (event.button != MouseButton::Left) {
                return false;
            }
            if (insideMute) {
                s.pressedMute = true;
                return true;
            }
            if (!inside) {
                return false;
            }
            s.pressed = true;
            s.starting = s.restarts();
            return true;
        case MouseEventType::Release: {
            if (event.button != MouseButton::Left) {
                return false;
            }
            if (s.pressedMute) {
                s.pressedMute = false;
                if (insideMute && s.config.onMute) {
                    s.config.onMute();
                }
                return true;
            }
            if (!s.pressed) {
                return false;
            }
            s.pressed = false;
            if (!inside) {
                return true;
            }
            if (s.starting) {
                if (s.config.onStart) {
                    s.config.onStart();
                }
            } else if (s.config.onStop) {
                s.config.onStop();
            }
            return true;
        }
        default:
            return false;
    }
}

std::optional<SurfaceCursor> Voice::cursorAt(const Extent2D<F32>& point) const {
    const bool insideMute = !impl->mute.empty() && impl->mute.contains(point.x, point.y);
    if (!insideMute && !impl->pill.contains(point.x, point.y)) {
        return std::nullopt;
    }
    return SurfaceCursor::Hand;
}

}  // namespace Jetstream::Sakura
