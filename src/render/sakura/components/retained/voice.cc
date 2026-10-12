#include <jetstream/render/sakura/components/retained/voice.hh>

#include <jetstream/logger.hh>
#include <jetstream/render/sakura/components/retained/button.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/render/tools/imgui_icons_ext.hh>

#include "../../context.hh"
#include "../../helpers.hh"
#include "../../retained/helpers.hh"
#include "orb.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <utility>

namespace Jetstream::Sakura::Retained {

namespace {

constexpr F32 kOrbRadiusRatio = 0.34f;
constexpr F32 kDismissSize = 22.0f;

struct VoiceBody : public Component {
    Voice::Config config;
    OrbView orb;
    Button dismiss;

    FrameClock clock;
    std::chrono::steady_clock::time_point epoch = std::chrono::steady_clock::now();
    F32 busyBlend = 0.0f;
    F32 activityBlend = 0.0f;
    F32 levelBlend = 0.0f;
    F32 hoverBlend = 0.0f;

    bool hovered = false;
    bool pressedOrb = false;
    Extent2D<F32> orbCenter = {0.0f, 0.0f};
    F32 orbRadius = 0.0f;
    Rect dismissRect;

    VoiceBody() {
        setClipsChildren(true);
        add(orb);
        add(dismiss);
    }

    void apply(Voice::Config next) {
        config = std::move(next);
    }

    bool insideOrb(const Extent2D<F32>& position) const {
        const F32 dx = position.x - orbCenter.x;
        const F32 dy = position.y - orbCenter.y;
        return dx * dx + dy * dy <= orbRadius * orbRadius;
    }

 protected:
    bool event(const MouseEvent& event) override {
        if (eventChildren(event)) {
            pressedOrb = false;
            return true;
        }
        const Extent2D<F32> position = {event.position.x, event.position.y};
        const bool inside = frame().contains(position.x, position.y);
        const bool overDismiss = hoverBlend > 0.5f && dismissRect.contains(position.x, position.y);
        const bool overOrb = !overDismiss && insideOrb(position);
        const bool startable = config.phase == Voice::Phase::Idle && config.onStart;

        switch (event.type) {
            case MouseEventType::Move:
                hovered = inside;
                if (overOrb && startable) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                }
                invalidate(Dirty::Paint);
                return false;
            case MouseEventType::Leave:
                hovered = false;
                pressedOrb = false;
                invalidate(Dirty::Paint);
                return false;
            case MouseEventType::Click:
                if (event.button != MouseButton::Left) {
                    return false;
                }
                pressedOrb = overOrb && startable;
                return pressedOrb;
            case MouseEventType::Release: {
                if (event.button != MouseButton::Left) {
                    return false;
                }
                const bool start = pressedOrb && overOrb && startable;
                pressedOrb = false;
                if (start) {
                    config.onStart();
                    return true;
                }
                return false;
            }
            default:
                return false;
        }
    }

    void layout(const Context& ctx) override {
        const auto now = std::chrono::steady_clock::now();
        const F32 dt = clock.tick(now, 0.016f);
        const F32 time = std::chrono::duration<F32>(now - epoch).count();

        const bool active = config.phase != Voice::Phase::Idle;
        const bool busy = config.phase == Voice::Phase::Busy;
        const bool engaged = busy || config.phase == Voice::Phase::Listening;
        const F32 level = active ? std::clamp(config.level, 0.0f, 1.0f) : 0.0f;
        busyBlend = Approach(busyBlend, busy ? 1.0f : 0.0f, dt / 0.45f);
        activityBlend = Approach(activityBlend, engaged ? 1.0f : (active ? 0.35f : 0.0f), dt / 0.6f);
        levelBlend = Approach(levelBlend, level, dt / (level > levelBlend ? 0.05f : 0.18f));
        if (!ctx.hovered) {
            hovered = false;
        }
        hoverBlend = Approach(hoverBlend, hovered ? 1.0f : 0.0f, dt / 0.08f);

        const Rect bounds = frame();
        const F32 ratio = ctx.pixelRatio;
        const bool visible = !bounds.empty();

        orbRadius = std::max(0.0f, std::min(bounds.width, bounds.height) * kOrbRadiusRatio);
        orbCenter = {bounds.x + bounds.width * 0.5f, bounds.y + bounds.height * 0.5f};
        orb.update({
            .id = jst::fmt::format("{}:orb", config.id),
            .rect = bounds,
            .orbCenter = orbCenter,
            .orbRadius = orbRadius,
            .visible = visible && orbRadius > 0.0f,
            .time = time,
            .busy = busyBlend,
            .activity = activityBlend,
            .level = levelBlend,
            .listeningColor = ctx.color("voice_orb_listening"),
            .busyColor = ctx.color("voice_orb_busy"),
            .listeningAccent = ctx.color("voice_orb_listening_accent"),
            .busyAccent = ctx.color("voice_orb_busy_accent"),
        });
        layoutChild(ctx, orb, bounds);

        const F32 badge = kDismissSize * ratio;
        const F32 diagonal = orbRadius * 0.7071f;
        dismissRect = {
            std::round(orbCenter.x + diagonal - badge * 0.5f),
            std::round(orbCenter.y - diagonal - badge * 0.5f),
            badge,
            badge,
        };

        const F32 alpha = std::clamp(hoverBlend, 0.0f, 1.0f);
        dismiss.update({
            .id = jst::fmt::format("{}:dismiss", config.id),
            .str = ICON_FA_XMARK,
            .disabled = alpha <= 0.5f,
            .colorKey = "card",
            .hoveredColorKey = "button_hovered",
            .activeColorKey = "button_active",
            .borderColorKey = "border",
            .textColorKey = "text_secondary",
            .disabledAlpha = 1.0f,
            .opacity = alpha,
            .fontSize = badge * 0.5f,
            .fontName = Typography::IconFont,
            .cornerRadius = badge * 0.5f,
            .borderWidth = 1.0f * ratio,
            .maxCharacters = 4,
            .onClick = [this]() {
                if (config.onDismiss) {
                    config.onDismiss();
                }
            },
        });
        layoutChild(ctx, dismiss, visible && alpha > 0.01f ? dismissRect : Rect{});
    }
};

}  // namespace

struct Voice::Impl {
    VoiceBody body;
};

Voice::Voice() {
    impl = std::make_unique<Impl>();
    setClipsChildren(true);
    add(impl->body);
}

Voice::~Voice() = default;

bool Voice::update(Config config) {
    impl->body.apply(std::move(config));
    return true;
}

void Voice::layout(const Context& ctx) {
    layoutChild(ctx, impl->body, frame());
}

}  // namespace Jetstream::Sakura::Retained
