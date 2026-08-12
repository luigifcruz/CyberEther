#include <jetstream/render/sakura/components/retained/button.hh>

#include <jetstream/render/sakura/components/retained/box.hh>
#include <jetstream/render/sakura/components/retained/label.hh>

#include "../../helpers.hh"
#include "../../retained/helpers.hh"
#include "../../retained/text_metrics.hh"

#include <algorithm>
#include <utility>

namespace Jetstream::Sakura::Retained {

namespace {

constexpr F32 kHorizontalPadding = 8.0f;
constexpr F32 kVerticalPadding = 4.0f;

}  // namespace

struct Button::Impl {
    Config config;
    Box box;
    Label label;
    mutable TextMetrics textMetrics;
    bool hovered = false;
    bool pressed = false;
};

Button::Button() {
    this->impl = std::make_unique<Impl>();
    setClipsChildren(true);
    add(this->impl->box);
    add(this->impl->label);
}

Button::~Button() = default;

bool Button::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

Extent2D<F32> Button::measure(const Context& ctx, Extent2D<F32> available) {
    impl->textMetrics.setWindow(ctx.render);

    const F32 ratio = ctx.pixelRatio;
    const F32 fontSizePixels = impl->config.fontSize;
    const F32 textWidth = impl->textMetrics.measure(impl->config.fontName, impl->config.str, fontSizePixels);

    const F32 width = textWidth + 2.0f * kHorizontalPadding * ratio;
    const F32 height = fontSizePixels + 2.0f * kVerticalPadding * ratio;

    return {std::min(width, available.x), std::min(height, available.y)};
}

void Button::layout(const Context& ctx) {
    const auto& c = impl->config;

    const Rect bounds = frame();
    const Rect rect = bounds;
    const Rect clipRect = Intersect(frame(), clip());
    const bool visible = !bounds.empty();
    const F32 fontSizePixels = c.fontSize;

    ColorRGBA<F32> bg = ctx.color(c.colorKey);
    ColorRGBA<F32> fg = ctx.color(c.textColorKey);
    ColorRGBA<F32> border = ctx.color(c.borderColorKey);

    if (c.disabled) {
        bg.a *= c.disabledAlpha;
        fg.a *= c.disabledAlpha;
        border.a *= c.disabledAlpha;
    } else if (impl->pressed) {
        bg = ctx.color(c.activeColorKey);
    } else if (impl->hovered) {
        bg = ctx.color(c.hoveredColorKey);
    }

    impl->box.update({
        .id = c.id + ":bg",
        .instances = {{.rect = rect, .visible = visible, .backgroundColor = bg}},
        .clip = clipRect,
        .cornerRadius = c.cornerRadius,
        .borderWidth = c.borderWidth,
        .borderColor = border,
    });

    impl->label.update({
        .id = c.id + ":label",
        .instances = {{
            .rect = rect,
            .str = c.str,
            .visible = visible,
            .color = fg,
            .fontSize = fontSizePixels,
            .alignment = {1, 1},
        }},
        .clip = clipRect,
        .fontName = c.fontName,
        .sharpness = 0.45f,
        .maxCharacters = c.maxCharacters,
    });

    layoutChild(ctx, impl->box, bounds);
    layoutChild(ctx, impl->label, bounds);

}

bool Button::event(const MouseEvent& event) {
    const auto setState = [&](bool nextHovered, bool nextPressed) {
        if (nextHovered == impl->hovered && nextPressed == impl->pressed) {
            return;
        }
        impl->hovered = nextHovered;
        impl->pressed = nextPressed;
        invalidate(Dirty::Paint);
    };

    if (impl->config.disabled) {
        setState(false, false);
        return false;
    }

    const bool inside = frame().contains(event.position.x, event.position.y);
    switch (event.type) {
        case MouseEventType::Move:
            if (inside) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            }
            setState(inside, impl->pressed && inside);
            return false;
        case MouseEventType::Leave:
            setState(false, false);
            return false;
        case MouseEventType::Click:
            if (event.button == MouseButton::Left && inside) {
                setState(true, true);
                return true;
            }
            return false;
        case MouseEventType::Release:
            if (event.button == MouseButton::Left && impl->pressed && inside) {
                setState(inside, false);
                if (impl->config.onClick) {
                    impl->config.onClick();
                }
                return true;
            }
            setState(inside, false);
            return false;
        default:
            return false;
    }
}

}  // namespace Jetstream::Sakura::Retained
