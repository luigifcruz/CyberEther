#include <jetstream/render/sakura/components/controls.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/dropdown.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/layout.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/components/text_grid.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/helpers.hh"
#include "../detail/options.hh"
#include "../detail/text_metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kInnerSpacing = 8.0f;
constexpr F32 kNaturalControlWidth = 200.0f;

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

F32 FontPx(const Context& ctx) {
    return Px(ctx, Typography::FontSize);
}

std::string Pick(const std::string& custom, const char* fallback) {
    return custom.empty() ? std::string(fallback) : custom;
}

}  // namespace

struct Button::Impl {
    Config config;
    Box frame;
    Box underline;
    Text label;
    bool hovered = false;
    bool pressed = false;

    bool textual() const {
        return config.variant == Variant::Text;
    }

    std::string textKey() const {
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
    addChild(impl->underline);
    addChild(impl->label);
}

Button::~Button() = default;

void Button::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    impl->label.update({
        .str = impl->config.str,
        .colorKey = impl->textKey(),
        .scale = impl->config.textScale,
    });
    if (impl->config.disabled) {
        impl->hovered = false;
        impl->pressed = false;
    }
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

    impl->frame.update({
        .instances = {{.rect = bounds, .visible = !impl->textual() && !bounds.empty(),
                       .backgroundColor = WithAlpha(ctx.color(background), alpha)}},
        .clip = bounds,
        .cornerRadius = Px(ctx, config.rounding),
        .borderWidth = Px(ctx, 1.0f),
        .borderColor = WithAlpha(ctx.color(Pick(config.borderColorKey, keys.border)), alpha),
        .strokeFringe = true,
    });

    const auto text = measureChild(impl->label, ctx, {std::numeric_limits<F32>::infinity(), bounds.height});
    const Rect textRect = impl->textual()
        ? Rect{bounds.x, bounds.y, text.x, text.y}
        : Rect{bounds.x + impl->padding(ctx).x +
                   std::floor(std::max(0.0f, bounds.width - 2.0f * impl->padding(ctx).x - text.x) * config.textAlign),
               bounds.y + std::floor((bounds.height - text.y) * 0.5f), text.x, text.y};
    layoutChild(ctx, impl->frame, bounds);
    layoutChild(ctx, impl->label, textRect);

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
                invalidate(Dirty::Layout);
            }
            return false;
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            if (impl->hovered || impl->pressed) {
                impl->hovered = false;
                impl->pressed = false;
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

struct TextInput::Impl {
    Config config;
    Box frame;
    TextGrid grid;
    Label hint;
    std::string buffer;
    std::string original;
    bool submitted = false;
    bool cancelled = false;
    std::string gridId;

    void focusChanged(bool focused) {
        if (focused) {
            original = config.value;
            return;
        }
        if (!cancelled && !submitted && config.submit == Submit::OnCommit && buffer != config.value &&
            config.onChange) {
            config.onChange(buffer);
        }
        if (cancelled && config.submit == Submit::OnEdit && buffer != original && config.onChange) {
            config.onChange(original);
        }
        grid.revert();
        buffer = config.value;
        submitted = false;
        cancelled = false;
    }
};

TextInput::TextInput() {
    impl = std::make_unique<Impl>();
    impl->gridId = jst::fmt::format("text-input:{}", static_cast<const void*>(this));
    setClipsChildren(true);
    addChild(impl->frame);
    addChild(impl->grid);
    addChild(impl->hint);
}

TextInput::~TextInput() = default;

void TextInput::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    const bool focused = impl->grid.focused();
    if (!focused) {
        impl->buffer = impl->config.value;
    }
    impl->grid.offerFocus(impl->config.focus && !focused, impl->config.selectAllOnFocus);
}

bool TextInput::focused() const {
    return impl->grid.focused();
}

Extent2D<F32> TextInput::measure(const Context& ctx, Extent2D<F32> available) {
    return {FillLength(available.x, Px(ctx, kNaturalControlWidth)),
            ResolveLength(ctx, impl->config.height, FontPx(ctx) + 2.0f * Px(ctx, impl->config.padding.y), available.y)};
}

void TextInput::layout(const Context& ctx) {
    auto& config = impl->config;
    const Rect bounds = frame();
    const bool focused = impl->grid.focused();

    const F32 fontSize = FontPx(ctx);
    const F32 padX = Px(ctx, config.padding.x);
    const F32 padY = config.multiline ? Px(ctx, config.padding.y) : std::max(0.0f, (bounds.height - fontSize) * 0.5f);
    impl->frame.update({
        .instances = {{.rect = bounds, .visible = !bounds.empty(), .backgroundColor = ctx.color(config.backgroundColorKey)}},
        .clip = bounds,
        .cornerRadius = Px(ctx, kFrameRounding),
    });

    impl->grid.update({
        .id = impl->gridId,
        .value = focused ? impl->buffer : config.value,
        .editable = true,
        .fontSize = fontSize,
        .lineHeight = 1.0f,
        .fontName = Typography::MonoFont,
        .monospace = true,
        .lineNumbers = false,
        .showActiveLine = false,
        .scrollbar = config.multiline,
        .wrap = TextGrid::Wrap::None,
        .padding = Padding{padX, padY, padX, padY},
        .visibleLineCapacity = config.multiline ? 64u : 4u,
        .backgroundColorKey = "transparent",
        .textColorKey = "text_primary",
        .selectionColorKey = "text_selected_bg",
        .selectionMatchColorKey = "transparent",
        .activeLineColorKey = "transparent",
        .cursorColorKey = "text_primary",
        .submitOnEnter = !config.multiline && (config.submit != Submit::OnEdit || config.onSubmit),
        .singleLine = !config.multiline,
        .selectAllOnFocus = config.selectAllOnFocus,
        .onChange = [this](std::string value) {
            impl->buffer = std::move(value);
            if (impl->config.submit == Submit::OnEdit && impl->config.onChange) {
                impl->config.onChange(impl->buffer);
            }
        },
        .onSubmit = [this](std::string value) {
            impl->buffer = std::move(value);
            impl->submitted = true;
            if (impl->config.submit != Submit::OnEdit && impl->config.onChange) {
                impl->config.onChange(impl->buffer);
            }
            if (impl->config.onSubmit) {
                impl->config.onSubmit(impl->buffer);
            }
            impl->grid.blur();
        },
        .onCancel = [this]() {
            impl->cancelled = true;
            if (impl->config.onCancel) {
                impl->config.onCancel();
            }
        },
        .onFocusChange = [this](bool focused) {
            impl->focusChanged(focused);
        },
    });

    impl->hint.update({
        .instances = {{
            .rect = {bounds.x + padX, bounds.y + padY, std::max(0.0f, bounds.width - 2.0f * padX), fontSize},
            .str = config.hint,
            .visible = impl->buffer.empty() && !config.hint.empty() && !bounds.empty(),
            .color = ctx.color("text_secondary"),
            .fontSize = fontSize,
            .alignment = {0, 1},
        }},
        .clip = bounds,
        .fontName = Typography::MonoFont,
        .maxCharacters = 128,
    });

    layoutChild(ctx, impl->frame, bounds);
    layoutChild(ctx, impl->grid, bounds);
    layoutChild(ctx, impl->hint, bounds);
}

bool TextInput::mouseEvent(const MouseEvent& event) {
    return mouseEventChildren(event);
}

std::optional<SurfaceCursor> TextInput::cursorAt(const Extent2D<F32>& point) const {
    if (!frame().contains(point.x, point.y)) {
        return std::nullopt;
    }
    return SurfaceCursor::TextInput;
}

struct Checkbox::Impl {
    Config config;
    Box square;
    Polyline mark;
    Text label;
    bool hovered = false;
    bool pressed = false;
};

Checkbox::Checkbox() {
    impl = std::make_unique<Impl>();
    addChild(impl->square);
    addChild(impl->mark);
    addChild(impl->label);
}

Checkbox::~Checkbox() = default;

void Checkbox::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->label.update({.str = config.label});
    impl->config = std::move(config);
}

Extent2D<F32> Checkbox::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 square = FrameHeightPx(ctx);
    F32 width = square;
    if (!impl->config.label.empty()) {
        const F32 infinite = std::numeric_limits<F32>::infinity();
        width += Px(ctx, kInnerSpacing) + measureChild(impl->label, ctx, {infinite, infinite}).x;
    }
    return FitExtent({width, square}, available);
}

void Checkbox::layout(const Context& ctx) {
    const Rect bounds = frame();
    const F32 square = std::min(bounds.height, FrameHeightPx(ctx));
    const Rect box = {bounds.x, bounds.y + std::floor((bounds.height - square) * 0.5f), square, square};
    const char* key = impl->pressed && impl->hovered ? "frame_bg_active" : impl->hovered ? "frame_bg_hovered" : "card";
    impl->square.update({
        .instances = {{.rect = box, .visible = !bounds.empty(), .backgroundColor = ctx.color(key)}},
        .clip = bounds,
        .cornerRadius = Px(ctx, kFrameRounding),
    });

    const F32 pad = std::max(1.0f, std::floor(square / 6.0f));
    F32 size = square - 2.0f * pad;
    const F32 thickness = std::max(size / 5.0f, 1.0f);
    size -= thickness * 0.5f;
    const F32 originX = box.x + pad + thickness * 0.25f;
    const F32 originY = box.y + pad + thickness * 0.25f;
    const F32 third = size / 3.0f;
    const F32 bx = originX + third;
    const F32 by = originY + size - third * 0.5f;
    impl->mark.update({
        .instances = {{
            .points = {{bx - third, by - third}, {bx, by}, {bx + third * 2.0f, by - third * 2.0f}},
            .color = ctx.color("accent_color"),
            .thickness = thickness,
            .visible = impl->config.value && !bounds.empty(),
        }},
        .clip = bounds,
        .fringe = Fringe(ctx),
        .capacity = 64,
    });

    const F32 labelX = box.right() + Px(ctx, kInnerSpacing);
    const F32 labelHeight = measureChild(impl->label, ctx, {std::numeric_limits<F32>::infinity(), bounds.height}).y;
    layoutChild(ctx, impl->square, bounds);
    layoutChild(ctx, impl->mark, bounds);
    layoutChild(ctx, impl->label, {labelX, box.y + std::floor((square - labelHeight) * 0.5f),
                                   std::max(0.0f, bounds.right() - labelX), labelHeight});
}

bool Checkbox::mouseEvent(const MouseEvent& event) {
    const bool inside = frame().contains(event.position.x, event.position.y);
    switch (event.type) {
        case MouseEventType::Move:
            if (inside != impl->hovered) {
                impl->hovered = inside;
                invalidate(Dirty::Layout);
            }
            return false;
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            if (impl->hovered || impl->pressed) {
                impl->hovered = false;
                impl->pressed = false;
                invalidate(Dirty::Layout);
            }
            return false;
        case MouseEventType::Click:
            if (!inside || event.button != MouseButton::Left) {
                return false;
            }
            impl->pressed = true;
            invalidate(Dirty::Layout);
            return true;
        case MouseEventType::Release:
            if (event.button != MouseButton::Left || !impl->pressed) {
                return false;
            }
            impl->pressed = false;
            invalidate(Dirty::Layout);
            if (inside && impl->config.onChange) {
                impl->config.onChange(!impl->config.value);
            }
            return true;
        default:
            return false;
    }
}

std::optional<SurfaceCursor> Checkbox::cursorAt(const Extent2D<F32>& point) const {
    if (!frame().contains(point.x, point.y)) {
        return std::nullopt;
    }
    return SurfaceCursor::Hand;
}

struct ProgressBar::Impl {
    Config config;
    Box background;
    Box fill;
    Label overlay;
    TextMetrics metrics;
};

ProgressBar::ProgressBar() {
    impl = std::make_unique<Impl>();
    addChild(impl->background);
    addChild(impl->fill);
    addChild(impl->overlay);
}

ProgressBar::~ProgressBar() = default;

void ProgressBar::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
}

Extent2D<F32> ProgressBar::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& size = impl->config.size;
    return {ResolveLength(ctx, size.x, Px(ctx, kNaturalControlWidth), available.x),
            ResolveLength(ctx, size.y, FrameHeightPx(ctx), available.y)};
}

void ProgressBar::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const F32 rounding = Px(ctx, kFrameRounding);
    const F32 fraction = std::clamp(config.value, 0.0f, 1.0f);
    const F32 fillEnd = bounds.x + bounds.width * fraction;
    impl->background.update({
        .instances = {{.rect = bounds, .visible = !bounds.empty(),
                       .backgroundColor = ctx.color(config.backgroundColorKey.empty() ? "card"
                                                                                      : config.backgroundColorKey)}},
        .clip = bounds,
        .cornerRadius = rounding,
    });
    impl->fill.update({
        .instances = {{.rect = bounds, .visible = !bounds.empty() && fraction > 0.0f,
                       .backgroundColor = ctx.color(config.colorKey)}},
        .clip = Intersect(bounds, {bounds.x, bounds.y, std::max(0.0f, fillEnd - bounds.x), bounds.height}),
        .cornerRadius = rounding,
    });
    impl->metrics.setWindow(ctx.render);
    const F32 fontSize = FontPx(ctx);
    const F32 textWidth = impl->metrics.measure(Typography::MonoFont, config.overlay, fontSize);
    const F32 textX = std::clamp(fillEnd + Px(ctx, Stack::Spacing), bounds.x,
                                 std::max(bounds.x, bounds.right() - textWidth - Px(ctx, kInnerSpacing)));
    impl->overlay.update({
        .instances = {{
            .rect = {textX, bounds.y, textWidth, bounds.height},
            .str = config.overlay,
            .visible = !bounds.empty() && !config.overlay.empty(),
            .color = ctx.color("text_primary"),
            .fontSize = fontSize,
            .alignment = {0, 1},
        }},
        .clip = bounds,
        .fontName = Typography::MonoFont,
        .maxCharacters = 128,
    });
    layoutChild(ctx, impl->background, bounds);
    layoutChild(ctx, impl->fill, bounds);
    layoutChild(ctx, impl->overlay, bounds);
}

struct Combo::Impl {
    Config config;
    Box frame;
    Polyline arrow;
    Label preview;
    Dropdown dropdown;
    bool hovered = false;
};

Combo::Combo() {
    impl = std::make_unique<Impl>();
    addChild(impl->frame);
    addChild(impl->arrow);
    addChild(impl->preview);
}

Combo::~Combo() = default;

void Combo::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    const auto& current = impl->config;
    if (current.disabled && impl->dropdown.isOpen()) {
        impl->dropdown.close();
    }
    impl->dropdown.update({
        .id = current.id,
        .options = current.options,
        .selected = SelectedOption(current),
        .padding = 8.0f,
        .rowPadding = {4.0f, 4.0f},
        .onSelect = OptionSelected(impl->config),
    });
}

Extent2D<F32> Combo::measure(const Context& ctx, Extent2D<F32> available) {
    return {ResolveLength(ctx, impl->config.width, Px(ctx, kNaturalControlWidth), available.x),
            FitLength(FrameHeightPx(ctx), available.y)};
}

void Combo::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const F32 alpha = config.disabled ? kDisabledAlpha : 1.0f;
    const F32 arrowSize = bounds.height;
    const F32 valueX2 = std::max(bounds.x, bounds.right() - arrowSize);
    const bool hasValue = valueX2 > bounds.x;
    const bool hovered = impl->hovered && !config.disabled;
    const auto frameColor = WithAlpha(ctx.color(hovered ? "frame_bg_hovered" : "card"), alpha);
    const auto buttonColor = WithAlpha(ctx.color(impl->dropdown.isOpen() || hovered ? "button_hovered" : "button"), alpha);
    impl->frame.update({
        .instances = {
            {.rect = {bounds.x, bounds.y, valueX2 - bounds.x, bounds.height}, .visible = !bounds.empty() && hasValue,
             .backgroundColor = frameColor, .corners = static_cast<U8>(Box::TopLeft | Box::BottomLeft)},
            {.rect = {valueX2, bounds.y, bounds.right() - valueX2, bounds.height}, .visible = !bounds.empty(),
             .backgroundColor = buttonColor,
             .corners = hasValue ? static_cast<U8>(Box::TopRight | Box::BottomRight) : Box::AllCorners},
        },
        .clip = bounds,
        .cornerRadius = Px(ctx, kFrameRounding),
    });

    const F32 fontSize = FontPx(ctx);
    const F32 padY = Px(ctx, kFramePaddingY);
    const F32 r = fontSize * 0.40f;
    const Extent2D<F32> center = {valueX2 + padY + fontSize * 0.5f, bounds.y + padY + fontSize * 0.5f};
    impl->arrow.update({
        .instances = {{
            .points = {
                {center.x, center.y + 0.75f * r},
                {center.x - 0.866f * r, center.y - 0.75f * r},
                {center.x + 0.866f * r, center.y - 0.75f * r},
            },
            .color = WithAlpha(ctx.color("text_primary"), alpha),
            .filled = true,
            .visible = !bounds.empty(),
        }},
        .clip = bounds,
        .fringe = Fringe(ctx),
    });

    const F32 padX = Px(ctx, kFramePaddingX);
    impl->preview.update({
        .instances = {{
            .rect = {bounds.x + padX, bounds.y + padY, std::max(0.0f, valueX2 - bounds.x - padX), fontSize},
            .str = config.value.empty() ? std::string("Select") : config.value,
            .visible = !bounds.empty(),
            .color = WithAlpha(ctx.color("text_primary"), alpha),
            .fontSize = fontSize,
            .alignment = {0, 1},
        }},
        .clip = Intersect(bounds, {bounds.x, bounds.y, std::max(0.0f, valueX2 - bounds.x), bounds.height}),
        .fontName = Typography::MonoFont,
        .maxCharacters = 128,
    });

    layoutChild(ctx, impl->frame, bounds);
    layoutChild(ctx, impl->arrow, bounds);
    layoutChild(ctx, impl->preview, bounds);

    impl->dropdown.request(ctx, bounds);
}

bool Combo::mouseEvent(const MouseEvent& event) {
    if (impl->config.disabled) {
        return false;
    }
    const bool inside = frame().contains(event.position.x, event.position.y);
    if (event.type == MouseEventType::Move && inside != impl->hovered) {
        impl->hovered = inside;
        invalidate(Dirty::Layout);
    } else if ((event.type == MouseEventType::Leave || event.type == MouseEventType::Cancel) && impl->hovered) {
        impl->hovered = false;
        invalidate(Dirty::Layout);
    }
    if (event.type == MouseEventType::Click && event.button == MouseButton::Left && inside) {
        if (impl->dropdown.isOpen()) {
            impl->dropdown.close();
        } else {
            impl->dropdown.open();
        }
        invalidate(Dirty::Layout);
        return true;
    }
    return false;
}

std::optional<SurfaceCursor> Combo::cursorAt(const Extent2D<F32>& point) const {
    if (impl->config.disabled || !frame().contains(point.x, point.y)) {
        return std::nullopt;
    }
    return SurfaceCursor::Hand;
}

struct SettingField::Impl {
    Config config;
    Text label;
    Text description;
    Divider divider;
    Stack stack;
};

SettingField::SettingField() {
    impl = std::make_unique<Impl>();
    addChild(impl->stack);
}

SettingField::~SettingField() = default;

void SettingField::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    impl->label.update({.str = impl->config.label});
    impl->description.update({
        .str = impl->config.description,
        .tone = Text::Tone::Disabled,
        .wrapped = true,
    });
    impl->divider.update({});
    std::vector<Stack::Item> items;
    if (!impl->config.label.empty()) {
        items.push_back({.child = &impl->label});
    }
    if (impl->config.control) {
        items.push_back({.child = impl->config.control});
    }
    if (!impl->config.description.empty()) {
        items.push_back({.child = &impl->description});
    }
    if (impl->config.divider) {
        items.push_back({.child = &impl->divider});
    }
    impl->stack.update({.items = std::move(items)});
}

Extent2D<F32> SettingField::measure(const Context& ctx, Extent2D<F32> available) {
    return measureChild(impl->stack, ctx, available);
}

void SettingField::layout(const Context& ctx) {
    layoutChild(ctx, impl->stack, frame());
}

struct QrCode::Impl {
    Config config;
    Box modules;
    Rect code;

    static constexpr int kBorder = 2;

    F32 side(const Context& ctx) const {
        if (config.width <= 0 || config.data.empty()) {
            return 0.0f;
        }
        return static_cast<F32>(config.width + 2 * kBorder) * Px(ctx, config.moduleSize);
    }
};

QrCode::QrCode() {
    impl = std::make_unique<Impl>();
    addChild(impl->modules);
}

QrCode::~QrCode() = default;

void QrCode::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    if (config.data == impl->config.data && config.width == impl->config.width &&
        config.moduleSize == impl->config.moduleSize) {
        impl->config.onClick = std::move(config.onClick);
        return;
    }
    impl->config = std::move(config);
}

Extent2D<F32> QrCode::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 side = impl->side(ctx);
    return FitExtent({std::isfinite(available.x) ? std::max(available.x, side) : side, side}, available);
}

void QrCode::layout(const Context& ctx) {
    const Rect bounds = frame();
    const auto& config = impl->config;
    const F32 side = impl->side(ctx);
    const F32 module = Px(ctx, config.moduleSize);
    const F32 x0 = std::floor(bounds.x + std::max(0.0f, (bounds.width - side) * 0.5f));
    const F32 y0 = std::floor(bounds.y);
    impl->code = {x0, y0, side, side};

    std::vector<Box::Instance> instances;
    if (side > 0.0f) {
        instances.push_back({.rect = impl->code, .backgroundColor = {1.0f, 1.0f, 1.0f, 1.0f}});
        for (int y = 0; y < config.width; ++y) {
            int x = 0;
            while (x < config.width) {
                const auto at = [&](int column) {
                    const U64 index = static_cast<U64>(y * config.width + column);
                    return index < config.data.size() && config.data[index] != 0;
                };
                if (!at(x)) {
                    ++x;
                    continue;
                }
                int end = x;
                while (end < config.width && at(end)) {
                    ++end;
                }
                instances.push_back({
                    .rect = {x0 + static_cast<F32>(x + Impl::kBorder) * module,
                             y0 + static_cast<F32>(y + Impl::kBorder) * module,
                             static_cast<F32>(end - x) * module, module},
                    .backgroundColor = {0.0f, 0.0f, 0.0f, 1.0f},
                });
                x = end;
            }
        }
    }
    impl->modules.update({.instances = std::move(instances), .clip = bounds});
    layoutChild(ctx, impl->modules, bounds);
}

bool QrCode::mouseEvent(const MouseEvent& event) {
    if (event.type != MouseEventType::Click || event.button != MouseButton::Left ||
        !impl->code.contains(event.position.x, event.position.y) || !impl->config.onClick) {
        return false;
    }
    impl->config.onClick();
    return true;
}

std::optional<SurfaceCursor> QrCode::cursorAt(const Extent2D<F32>& point) const {
    if (!impl->code.contains(point.x, point.y)) {
        return std::nullopt;
    }
    return SurfaceCursor::Hand;
}

}  // namespace Jetstream::Sakura
