#include <jetstream/render/sakura/components/pill_input.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/card.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kPillInputPadX = 14.0f;
constexpr F32 kPillInputIcon = 13.0f;
constexpr F32 kPillInputText = 14.0f;

}  // namespace

struct PillInput::Impl {
    Config config;
    Card card;
    Box ring;
    Text icon;
    TextInput input;
    bool pressed = false;
};

PillInput::PillInput() {
    impl = std::make_unique<Impl>();
    setClipsChildren(false);
    mountChildren({&impl->card, &impl->ring, &impl->icon, &impl->input});
}

PillInput::~PillInput() = default;

void PillInput::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    const auto& current = impl->config;
    if (current.disabled) {
        impl->pressed = false;
    }
    impl->icon.update({
        .str = current.icon,
        .colorKey = "text_secondary",
        .scale = kPillInputIcon / Typography::FontSize,
        .opacity = current.disabled ? kDisabledAlpha : 1.0f,
    });
    impl->input.update({
        .value = current.value,
        .hint = current.hint,
        .submit = current.submit,
        .password = current.password,
        .disabled = current.disabled,
        .onChange = [this](const std::string& value) {
            if (impl->config.onChange) {
                impl->config.onChange(value);
            }
        },
        .backgroundColorKey = "transparent",
        .padding = {0.0f, 0.0f},
        .fontSize = kPillInputText,
        .fontName = Typography::DisplayFont,
    });
}

Extent2D<F32> PillInput::measure(const Context& ctx, Extent2D<F32> available) {
    return {ResolveLength(ctx, impl->config.width, Px(ctx, kNaturalControlWidth), available.x),
            FitLength(Px(ctx, kPillHeight), available.y)};
}

void PillInput::layout(const Context& ctx) {
    auto& s = *impl;
    const Rect bounds = frame();
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const F32 radius = bounds.height * 0.5f;
    const F32 alpha = s.config.disabled ? kDisabledAlpha : 1.0f;
    s.card.update({
        .rects = {bounds},
        .clip = clip(),
        .cornerRadius = radius,
        .softness = Px(ctx, kPillShadowSoftness),
        .drop = Px(ctx, kPillShadowDrop),
        .borderWidth = Px(ctx, kPillBorder),
        .lineWidth = Px(ctx, 1.0f),
        .fillKey = kPillFill,
        .opacity = alpha,
    });
    layoutChild(ctx, s.card, bounds);
    s.ring.update({
        .instances = {{.rect = bounds, .visible = !bounds.empty() && s.input.focused(),
                       .backgroundColor = {0.0f, 0.0f, 0.0f, 0.0f}}},
        .cornerRadius = radius,
        .borderWidth = Px(ctx, 1.0f),
        .borderColor = ctx.color("action_btn_outline"),
        .strokeFringe = true,
        .capacity = 1,
    });
    layoutChild(ctx, s.ring, bounds);

    F32 x = bounds.x + Px(ctx, kPillInputPadX);
    s.icon.setVisible(!s.config.icon.empty());
    if (!s.config.icon.empty()) {
        const Extent2D<F32> size = measureChild(s.icon, ctx, {infinite, infinite});
        layoutChild(ctx, s.icon, {x, std::round(bounds.y + (bounds.height - size.y) * 0.5f), size.x, size.y}, bounds);
        x += size.x + Px(ctx, kPillIconGap);
    }
    const F32 right = bounds.right() - Px(ctx, kPillInputPadX);
    layoutChild(ctx, s.input, {x, bounds.y, std::max(0.0f, right - x), bounds.height}, bounds);
}

bool PillInput::mouseEvent(const MouseEvent& event) {
    auto& s = *impl;
    if (mouseEventChildren(event)) {
        return true;
    }
    if (s.config.disabled) {
        return false;
    }
    const bool inside = frame().contains(event.position.x, event.position.y);
    switch (event.type) {
        case MouseEventType::Click:
            s.pressed = inside && event.button == MouseButton::Left;
            return inside;
        case MouseEventType::Release: {
            const bool pressed = s.pressed;
            s.pressed = false;
            if (pressed && inside && event.button == MouseButton::Left) {
                s.input.focus();
                invalidate(Dirty::Layout);
            }
            return pressed;
        }
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            s.pressed = false;
            return false;
        default:
            return false;
    }
}

std::optional<SurfaceCursor> PillInput::cursorAt(const Extent2D<F32>& point) const {
    if (impl->config.disabled || !frame().contains(point.x, point.y)) {
        return std::nullopt;
    }
    return SurfaceCursor::TextInput;
}

}  // namespace Jetstream::Sakura
