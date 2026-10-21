#include <jetstream/render/sakura/components/text_input.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/text_grid.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>

namespace Jetstream::Sakura {

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
    if (impl->config.disabled && impl->grid.focused()) {
        impl->grid.blur();
    }
    const bool focused = impl->grid.focused();
    if (!focused) {
        impl->buffer = impl->config.value;
    }
    impl->grid.offerFocus(impl->config.focus && !impl->config.disabled && !focused, impl->config.selectAllOnFocus);
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
    const F32 alpha = config.disabled ? kDisabledAlpha : 1.0f;

    const F32 fontSize = FontPx(ctx);
    const F32 padX = Px(ctx, config.padding.x);
    const F32 padY = config.multiline ? Px(ctx, config.padding.y) : std::max(0.0f, (bounds.height - fontSize) * 0.5f);
    impl->frame.update({
        .instances = {{.rect = bounds,
                       .visible = !bounds.empty(),
                       .backgroundColor = WithAlpha(ctx.color(config.backgroundColorKey), alpha)}},
        .clip = bounds,
        .cornerRadius = Px(ctx, config.rounding),
        .borderWidth = config.borderColorKey.empty() ? 0.0f : Px(ctx, 1.0f),
        .borderColor = WithAlpha(ctx.color(config.borderColorKey), alpha),
        .strokeFringe = !config.borderColorKey.empty(),
    });

    impl->grid.update({
        .id = impl->gridId,
        .value = focused ? impl->buffer : config.value,
        .editable = !config.disabled,
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
        .textColorKey = config.disabled ? "text_disabled" : "text_primary",
        .selectionColorKey = "text_selected_bg",
        .selectionMatchColorKey = "transparent",
        .activeLineColorKey = "transparent",
        .cursorColorKey = "text_primary",
        .submitOnEnter = !config.multiline && (config.submit != Submit::OnEdit || config.onSubmit),
        .singleLine = !config.multiline,
        .selectAllOnFocus = config.selectAllOnFocus,
        .masked = config.password,
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
            .color = WithAlpha(ctx.color("text_secondary"), alpha),
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
    if (impl->config.disabled) {
        return false;
    }
    return mouseEventChildren(event);
}

std::optional<SurfaceCursor> TextInput::cursorOverChildren(const Extent2D<F32>& point) const {
    if (!impl->config.disabled || !frame().contains(point.x, point.y)) {
        return std::nullopt;
    }
    return SurfaceCursor::Default;
}

std::optional<SurfaceCursor> TextInput::cursorAt(const Extent2D<F32>& point) const {
    if (impl->config.disabled || !frame().contains(point.x, point.y)) {
        return std::nullopt;
    }
    return SurfaceCursor::TextInput;
}

}  // namespace Jetstream::Sakura
