#include <jetstream/render/sakura/components/button.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/card.hh>
#include <jetstream/render/sakura/components/text.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace Jetstream::Sakura {

namespace {

struct ButtonKeys {
    const char* color;
    const char* hovered;
    const char* active;
    const char* text;
    const char* border;
};

ButtonKeys VariantKeys(Button::Variant variant) {
    switch (variant) {
        case Button::Variant::Action:
            return {"action_btn", "action_btn_hovered", "action_btn_active", "action_btn_text", "action_btn_outline"};
        case Button::Variant::Destructive:
            return {"destructive_btn", "destructive_btn_hovered", "destructive_btn_active", "destructive_btn_text",
                    "destructive_btn_outline"};
        case Button::Variant::Default:
        case Button::Variant::Text:
            break;
    }
    return {"button", "button_hovered", "button_active", "button_text", "button_outline"};
}

std::string Pick(const std::string& custom, const char* fallback) {
    return custom.empty() ? std::string(fallback) : custom;
}

}  // namespace

struct Button::Impl {
    Config config;
    Box frame;
    Card card;
    Box underline;
    Text label;
    bool hovered = false;
    bool pressed = false;

    bool textual() const {
        return config.variant == Variant::Text;
    }

    void pushLabel() {
        label.update({
            .str = config.str,
            .font = config.font,
            .colorKey = textKey(),
            .scale = config.textScale,
            .opacity = config.disabled && !textual() ? kDisabledAlpha : 1.0f,
        });
    }

    std::string textKey() const {
        if (hovered && !config.disabled && !config.hoveredTextColorKey.empty()) {
            return config.hoveredTextColorKey;
        }
        return textual() ? Pick(config.textColorKey, config.disabled ? "text_disabled" : "text_primary")
                         : Pick(config.textColorKey, VariantKeys(config.variant).text);
    }

    Extent2D<F32> padding(const Context& ctx) const {
        if (textual()) {
            return {0.0f, 0.0f};
        }
        return {Px(ctx, config.padding.x), Px(ctx, config.padding.y)};
    }
};

Button::Button() {
    impl = std::make_unique<Impl>();
    addChild(impl->frame);
    addChild(impl->card);
    addChild(impl->underline);
    addChild(impl->label);
}

Button::~Button() = default;

void Button::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    if (impl->config.disabled) {
        impl->hovered = false;
        impl->pressed = false;
    }
    impl->pushLabel();
}

Extent2D<F32> Button::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& config = impl->config;
    const auto text = measureChild(impl->label, ctx, {std::numeric_limits<F32>::infinity(), available.y});
    const auto pad = impl->padding(ctx);
    return {ResolveLength(ctx, config.size.x, text.x + 2.0f * pad.x, available.x),
            ResolveLength(ctx, config.size.y, text.y + 2.0f * pad.y, available.y)};
}

void Button::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const auto keys = VariantKeys(config.variant);
    const F32 alpha = config.disabled ? kDisabledAlpha : 1.0f;

    const std::string background = impl->pressed && impl->hovered ? Pick(config.activeColorKey, keys.active)
                                   : impl->hovered                 ? Pick(config.hoveredColorKey, keys.hovered)
                                                                   : Pick(config.colorKey, keys.color);
    const auto textColor = WithAlpha(ctx.color(impl->textKey()), alpha);

    const bool chrome = config.chrome && !impl->textual();
    impl->frame.setVisible(!chrome);
    impl->frame.update({
        .instances = {{.rect = bounds, .visible = !impl->textual() && !bounds.empty(),
                       .backgroundColor = WithAlpha(ctx.color(background), alpha)}},
        .clip = bounds,
        .cornerRadius = Px(ctx, config.rounding),
        .borderWidth = Px(ctx, 1.0f),
        .borderColor = WithAlpha(ctx.color(Pick(config.borderColorKey, keys.border)), alpha),
        .strokeFringe = true,
    });
    impl->card.setVisible(chrome);
    impl->card.update({
        .rects = {bounds},
        .clip = clip(),
        .cornerRadius = Px(ctx, config.rounding),
        .borderWidth = Px(ctx, 1.0f),
        .lineWidth = Px(ctx, 1.0f),
        .fillKey = background,
        .opacity = alpha,
        .capacity = 1,
    });

    const auto text = measureChild(impl->label, ctx, {std::numeric_limits<F32>::infinity(), bounds.height});
    const Rect textRect = impl->textual()
        ? Rect{bounds.x, bounds.y, text.x, text.y}
        : Rect{bounds.x + impl->padding(ctx).x +
                   std::floor(std::max(0.0f, bounds.width - 2.0f * impl->padding(ctx).x - text.x) * config.textAlign),
               bounds.y + std::floor((bounds.height - text.y) * 0.5f), text.x, text.y};
    layoutChild(ctx, impl->frame, bounds);
    layoutChild(ctx, impl->card, bounds);
    layoutChild(ctx, impl->label, textRect, bounds);

    const F32 line = Hairline(ctx);
    impl->underline.update({
        .instances = {{.rect = {textRect.x, textRect.bottom(), textRect.width, line},
                       .visible = impl->textual() && impl->hovered && !config.disabled,
                       .backgroundColor = textColor}},
        .clip = bounds,
    });
    layoutChild(ctx, impl->underline, bounds);
}

bool Button::mouseEvent(const MouseEvent& event) {
    if (impl->config.disabled) {
        return false;
    }
    const bool inside = frame().contains(event.position.x, event.position.y);
    switch (event.type) {
        case MouseEventType::Move:
            if (inside != impl->hovered) {
                impl->hovered = inside;
                impl->pushLabel();
                invalidate(Dirty::Layout);
            }
            return false;
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            if (impl->hovered || impl->pressed) {
                impl->hovered = false;
                impl->pressed = false;
                impl->pushLabel();
                invalidate(Dirty::Layout);
            }
            return false;
        case MouseEventType::Click:
            if (!inside || event.button != MouseButton::Left) {
                return false;
            }
            if (event.clicks >= 2 && impl->config.onDoubleClick) {
                impl->pressed = false;
                invalidate(Dirty::Layout);
                impl->config.onDoubleClick();
                return true;
            }
            impl->pressed = true;
            invalidate(Dirty::Layout);
            return true;
        case MouseEventType::Release: {
            if (event.button != MouseButton::Left || !impl->pressed) {
                return false;
            }
            impl->pressed = false;
            invalidate(Dirty::Layout);
            if (inside && impl->config.onClick) {
                impl->config.onClick();
            }
            return true;
        }
        default:
            return false;
    }
}

std::optional<SurfaceCursor> Button::cursorAt(const Extent2D<F32>& point) const {
    if (impl->config.disabled || !frame().contains(point.x, point.y)) {
        return std::nullopt;
    }
    return SurfaceCursor::Hand;
}

void NavigationItem::update(Config config) {
    Button::update({
        .str = std::move(config.label),
        .size = {Fill, 0.0f},
        .colorKey = config.selected ? "header_active" : "card",
        .hoveredColorKey = config.selected ? "header_active" : "header_hovered",
        .activeColorKey = "header_active",
        .borderColorKey = config.selected ? "accent_color" : "border",
        .textColorKey = config.selected ? "text_primary" : "text_secondary",
        .rounding = 10.0f,
        .padding = Extent2D<F32>{12.0f, 10.0f},
        .textAlign = 0.0f,
        .onClick = std::move(config.onSelect),
        .onDoubleClick = std::move(config.onDoubleClick),
    });
}

}  // namespace Jetstream::Sakura
