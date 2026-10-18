#include <jetstream/render/sakura/components/node/inputs.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/dropdown.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/text_grid.hh>
#include <jetstream/render/tools/imgui_icons_ext.hh>

#include "../../detail/options.hh"
#include "common.hh"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <utility>

namespace Jetstream::Sakura {

using namespace NodeStyle;

namespace {

std::string Trim(const std::string& value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

bool DecimalCharacter(U32 character) {
    return (character >= '0' && character <= '9') || character == '.' || character == '+' || character == '-';
}

bool ScientificCharacter(U32 character) {
    return DecimalCharacter(character) || character == 'e' || character == 'E';
}

std::optional<F64> ParseFloat(const std::string& text) {
    const std::string trimmed = Trim(text);
    if (trimmed.empty()) {
        return std::nullopt;
    }
    char* end = nullptr;
    errno = 0;
    const F64 value = std::strtod(trimmed.c_str(), &end);
    if (end == trimmed.c_str() || errno == ERANGE || !std::isfinite(value)) {
        return std::nullopt;
    }
    return value;
}

std::optional<I64> ParseInt(const std::string& text) {
    const std::string trimmed = Trim(text);
    if (trimmed.empty()) {
        return std::nullopt;
    }
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(trimmed.c_str(), &end, 10);
    if (errno == ERANGE) {
        return std::nullopt;
    }
    if (end == trimmed.c_str()) {
        const auto asFloat = ParseFloat(trimmed);
        constexpr F64 lowest = static_cast<F64>(std::numeric_limits<I64>::min());
        constexpr F64 highest = static_cast<F64>(std::numeric_limits<I64>::max());
        if (!asFloat.has_value() || *asFloat < lowest || *asFloat >= highest) {
            return std::nullopt;
        }
        return static_cast<I64>(*asFloat);
    }
    return static_cast<I64>(value);
}

std::optional<U64> ParseUInt(const std::string& text) {
    const std::string trimmed = Trim(text);
    if (trimmed.empty() || trimmed.front() == '-') {
        return std::nullopt;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(trimmed.c_str(), &end, 10);
    if (end == trimmed.c_str() || errno == ERANGE) {
        return std::nullopt;
    }
    return static_cast<U64>(value);
}

std::string FormatFloat(F32 value, I32 precision) {
    return jst::fmt::format("{:.{}f}", value, std::clamp(precision, 0, 12));
}

std::string FormatGeneral(F32 value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.6g", static_cast<double>(value));
    return buffer;
}

struct Pressable {
    bool pressed = false;

    bool click(const MouseEvent& event, const Rect& rect) {
        if (event.type == MouseEventType::Click && event.button == MouseButton::Left &&
            rect.contains(event.position.x, event.position.y)) {
            pressed = true;
            return true;
        }
        return false;
    }

    bool release(const MouseEvent& event, const Rect& rect, bool requireInside = true) {
        if (event.type == MouseEventType::Cancel || event.type == MouseEventType::Leave) {
            pressed = false;
            return false;
        }
        if (event.type != MouseEventType::Release || event.button != MouseButton::Left || !pressed) {
            return false;
        }
        pressed = false;
        return !requireInside || rect.contains(event.position.x, event.position.y);
    }
};

}  // namespace

struct NodeTextInput::Impl {
    Config config;
    Box frame;
    TextGrid grid;
    Label overlay;
    TextMetrics metrics;
    std::string buffer;
    bool submitted = false;
    bool cancelled = false;

    void focusChanged(bool focused) {
        if (focused) {
            return;
        }
        if (!submitted && !cancelled && config.submit == Submit::OnCommit && buffer != config.value &&
            config.onChange) {
            config.onChange(buffer);
        }
        grid.revert();
        buffer = config.value;
        submitted = false;
        cancelled = false;
        if (config.onDeactivate) {
            config.onDeactivate();
        }
    }
};

NodeTextInput::NodeTextInput() {
    this->impl = std::make_unique<Impl>();
    setClipsChildren(true);
    addChild(impl->frame);
    addChild(impl->grid);
    addChild(impl->overlay);
}

NodeTextInput::~NodeTextInput() = default;

void NodeTextInput::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    if (!impl->grid.focused()) {
        impl->buffer = impl->config.value;
    }
}

bool NodeTextInput::focused() const {
    return impl->grid.focused();
}

void NodeTextInput::focus(bool selectAll) {
    impl->grid.focus(selectAll);
}

void NodeTextInput::focusWhenPlaced(bool selectAll) {
    impl->grid.offerFocus(true, selectAll);
}

Extent2D<F32> NodeTextInput::measure(const Context& ctx, Extent2D<F32> available) {
    return {FillLength(available.x), ResolveLength(ctx, impl->config.height, Px(ctx, FrameHeight(ctx)), available.y)};
}

void NodeTextInput::layout(const Context& ctx) {
    auto& config = impl->config;
    const Rect bounds = frame();

    const bool focused = impl->grid.focused();

    impl->metrics.setWindow(ctx.render);

    const F32 fontSize = Px(ctx, FontOf(ctx));
    const F32 padX = Px(ctx, FramePadX);
    const F32 padY = std::max(0.0f, (bounds.height - fontSize) * 0.5f);

    impl->frame.update({
        .instances = {{
            .rect = Snap(bounds),
            .visible = config.background && !bounds.empty(),
            .backgroundColor = ctx.color("card"),
        }},
        .clip = bounds,
        .cornerRadius = Px(ctx, FrameRounding),
    });

    impl->grid.update({
        .id = config.id + ":grid",
        .value = focused ? impl->buffer : config.value,
        .editable = true,
        .fontSize = fontSize,
        .lineHeight = 1.0f,
        .fontName = MonoFont,
        .monospace = true,
        .lineNumbers = false,
        .showActiveLine = false,
        .scrollbar = false,
        .wrap = TextGrid::Wrap::None,
        .padding = Padding{padX, padY, padX, padY},
        .visibleLineCapacity = 4,
        .backgroundColorKey = "transparent",
        .textColorKey = config.textColorKey,
        .selectionColorKey = "text_selected_bg",
        .selectionMatchColorKey = "transparent",
        .activeLineColorKey = "transparent",
        .cursorColorKey = "text_primary",
        .submitOnEnter = true,
        .singleLine = true,
        .selectAllOnFocus = config.selectAllOnFocus,
        .characterFilter = [this](U32 character) {
            return !impl->config.characterFilter || impl->config.characterFilter(character);
        },
        .onChange = [this](std::string value) {
            impl->buffer = std::move(value);
        },
        .onSubmit = [this](std::string value) {
            impl->buffer = std::move(value);
            impl->submitted = true;
            if (impl->config.onChange) {
                impl->config.onChange(impl->buffer);
            }
            impl->grid.blur();
        },
        .onCancel = [this]() {
            impl->cancelled = true;
        },
        .onFocusChange = [this](bool focused) {
            impl->focusChanged(focused);
        },
    });

    const F32 unitWidth = impl->metrics.measure(MonoFont, config.unit, fontSize);
    const Rect lineRect = {bounds.x + padX, bounds.y + padY, std::max(0.0f, bounds.width - 2.0f * padX), fontSize};
    const Rect unitRect = {bounds.right() - Px(ctx, UnitInset) - unitWidth, bounds.y, unitWidth, bounds.height};
    impl->overlay.update({
        .instances = {
            {
                .rect = lineRect,
                .str = config.hint,
                .visible = impl->buffer.empty() && !config.hint.empty() && !bounds.empty(),
                .color = ctx.color("text_secondary"),
                .fontSize = fontSize,
                .alignment = {0, 1},
            },
            {
                .rect = unitRect,
                .str = config.unit,
                .visible = !config.unit.empty() && !bounds.empty(),
                .color = ctx.color("text_secondary"),
                .fontSize = fontSize,
                .alignment = {0, 1},
            },
        },
        .clip = bounds,
        .fontName = MonoFont,
        .maxCharacters = 128,
    });

    layoutChild(ctx, impl->frame, bounds);
    layoutChild(ctx, impl->grid, bounds);
    layoutChild(ctx, impl->overlay, bounds);
}

bool NodeTextInput::mouseEvent(const MouseEvent& event) {
    return mouseEventChildren(event);
}

struct NodeFloatInput::Impl {
    Config config;
    Box background;
    NodeTextInput input;
    NodeTextInput stepInput;
    Label labels;
    Rect minusRect;
    Rect toggleRect;
    Rect plusRect;
    Pressable minus;
    Pressable toggle;
    Pressable plus;
    Pointer pointer;
    bool stepEditing = false;

    bool stepped() const {
        return config.step.has_value();
    }
};

NodeFloatInput::NodeFloatInput() {
    this->impl = std::make_unique<Impl>();
    addChild(impl->input);
    impl->stepInput.setVisible(false);
}

NodeFloatInput::~NodeFloatInput() = default;

void NodeFloatInput::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    const auto& current = impl->config;
    impl->input.update({
        .id = current.id + ":value",
        .value = FormatFloat(current.value, current.precision),
        .unit = current.unit,
        .submit = NodeTextInput::Submit::OnEnter,
        .selectAllOnFocus = true,
        .characterFilter = ScientificCharacter,
        .onChange = [this](const std::string& text) {
            const auto parsed = ParseFloat(text);
            if (!parsed.has_value()) {
                return;
            }
            const F32 next = static_cast<F32>(*parsed);
            if (next != impl->config.value && impl->config.onChange) {
                impl->config.onChange(next);
            }
        },
    });
    if (impl->stepped()) {
        impl->stepInput.update({
            .id = current.id + ":step",
            .value = FormatFloat(current.step.value_or(0.0f), current.precision),
            .submit = NodeTextInput::Submit::OnEnter,
            .selectAllOnFocus = true,
            .characterFilter = ScientificCharacter,
            .textColorKey = "text_secondary",
            .onChange = [this](const std::string& text) {
                const auto parsed = ParseFloat(text);
                if (parsed.has_value() && impl->config.onStepChange) {
                    impl->config.onStepChange(static_cast<F32>(*parsed));
                }
            },
            .onDeactivate = [this]() {
                impl->stepEditing = false;
                impl->stepInput.setVisible(false);
            },
        });
        setChildren({&impl->background, &impl->input, &impl->stepInput, &impl->labels});
    } else {
        setChildren({&impl->input});
    }
    const bool showStepInput = impl->stepped() && impl->stepEditing;
    if (impl->stepInput.visible() != showStepInput) {
        impl->stepInput.setVisible(showStepInput);
    }
}

Extent2D<F32> NodeFloatInput::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 height = impl->stepped() ? FrameHeight(ctx) + FontOf(ctx) + 4.0f : FrameHeight(ctx);
    return {FillLength(available.x), FitLength(Px(ctx, height), available.y)};
}

void NodeFloatInput::layout(const Context& ctx) {
    const Rect bounds = frame();
    const Rect clipRect = Intersect(bounds, clip());
    const F32 fontSize = Px(ctx, FontOf(ctx));

    if (!impl->stepped()) {
        layoutChild(ctx, impl->input, bounds);
        return;
    }

    const F32 inputHeight = Px(ctx, FrameHeight(ctx));
    const F32 buttonHeight = Px(ctx, FontOf(ctx) + 4.0f);
    const F32 buttonWidth = bounds.width * 0.2f;
    const F32 toggleWidth = bounds.width - buttonWidth * 2.0f;
    const F32 rowY = bounds.y + inputHeight;
    impl->minusRect = {bounds.x, rowY, buttonWidth, buttonHeight};
    impl->toggleRect = {bounds.x + buttonWidth, rowY, toggleWidth, buttonHeight};
    impl->plusRect = {bounds.x + buttonWidth + toggleWidth, rowY, buttonWidth, buttonHeight};

    impl->background.update({
        .instances = {{
            .rect = Snap({bounds.x, bounds.y, bounds.width, inputHeight + buttonHeight}),
            .visible = !bounds.empty(),
            .backgroundColor = ctx.color("card"),
        }},
        .clip = bounds,
        .cornerRadius = Px(ctx, FrameRounding),
    });

    const auto hoveredRect = [&](const Rect& rect) {
        return impl->pointer.inside(Intersect(rect, clipRect));
    };
    const bool minusHovered = hoveredRect(impl->minusRect);
    const bool toggleHovered = !impl->stepEditing && hoveredRect(impl->toggleRect);
    const bool plusHovered = hoveredRect(impl->plusRect);

    const auto labelColor = [&](bool hovered) {
        return ctx.color(hovered ? "text_primary" : "text_secondary");
    };
    impl->labels.update({
        .instances = {
            {.rect = impl->minusRect, .str = "-", .visible = true, .color = labelColor(minusHovered),
             .fontSize = fontSize, .alignment = {1, 1}},
            {.rect = impl->toggleRect, .str = "Step Size", .visible = !impl->stepEditing,
             .color = labelColor(toggleHovered), .fontSize = fontSize, .alignment = {1, 1}},
            {.rect = impl->plusRect, .str = "+", .visible = true, .color = labelColor(plusHovered),
             .fontSize = fontSize, .alignment = {1, 1}},
        },
        .clip = bounds,
        .fontName = MonoFont,
        .maxCharacters = 32,
        .capacity = 3,
    });

    layoutChild(ctx, impl->background, bounds);
    layoutChild(ctx, impl->input, {bounds.x, bounds.y, bounds.width, inputHeight});
    if (impl->stepInput.visible()) {
        layoutChild(ctx, impl->stepInput, impl->toggleRect);
    }
    layoutChild(ctx, impl->labels, bounds);
}

std::optional<SurfaceCursor> NodeFloatInput::cursorAt(const Extent2D<F32>& point) const {
    if (!impl->stepped()) {
        return std::nullopt;
    }
    const auto over = [&](const Rect& rect) {
        return Intersect(Intersect(rect, frame()), clip()).contains(point.x, point.y);
    };
    if (over(impl->minusRect) || (!impl->stepEditing && over(impl->toggleRect)) || over(impl->plusRect)) {
        return SurfaceCursor::Hand;
    }
    return std::nullopt;
}

bool NodeFloatInput::mouseEvent(const MouseEvent& event) {
    const Rect area = Intersect(frame(), clip());
    const bool stepped = impl->stepped();
    const bool moved = impl->pointer.track(event, area, {impl->minusRect, impl->toggleRect, impl->plusRect});
    if (moved && stepped) {
        invalidate(Dirty::Layout);
    }
    if (mouseEventChildren(event)) {
        return true;
    }
    if (!impl->stepped()) {
        return false;
    }

    auto& config = impl->config;
    switch (event.type) {
        case MouseEventType::Click:
            if (impl->minus.click(event, impl->minusRect) || impl->plus.click(event, impl->plusRect)) {
                return true;
            }
            if (!impl->stepEditing && impl->toggle.click(event, impl->toggleRect)) {
                return true;
            }
            return false;
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            impl->minus.release(event, impl->minusRect);
            impl->plus.release(event, impl->plusRect);
            impl->toggle.release(event, impl->toggleRect);
            return false;
        case MouseEventType::Release: {
            const F32 step = config.step.value_or(0.0f);
            if (impl->minus.release(event, impl->minusRect, false)) {
                if (config.onChange) {
                    config.onChange(config.value - step);
                }
                return true;
            }
            if (impl->plus.release(event, impl->plusRect, false)) {
                if (config.onChange) {
                    config.onChange(config.value + step);
                }
                return true;
            }
            if (impl->toggle.release(event, impl->toggleRect, false)) {
                impl->stepEditing = true;
                impl->stepInput.setVisible(true);
                impl->stepInput.focusWhenPlaced(false);
                return true;
            }
            return false;
        }
        default:
            return false;
    }
}

struct NodeIntInput::Impl {
    Config config;
    NodeTextInput input;
};

NodeIntInput::NodeIntInput() {
    this->impl = std::make_unique<Impl>();
    addChild(impl->input);
}

NodeIntInput::~NodeIntInput() = default;

void NodeIntInput::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    impl->input.update({
        .id = impl->config.id + ":value",
        .value = jst::fmt::format("{}", impl->config.value),
        .unit = impl->config.unit,
        .submit = NodeTextInput::Submit::OnEnter,
        .selectAllOnFocus = true,
        .characterFilter = DecimalCharacter,
        .onChange = [this](const std::string& text) {
            const auto parsed = ParseInt(text);
            if (parsed.has_value() && *parsed != impl->config.value && impl->config.onChange) {
                impl->config.onChange(*parsed);
            }
        },
    });
}

Extent2D<F32> NodeIntInput::measure(const Context& ctx, Extent2D<F32> available) {
    return {FillLength(available.x), FitLength(Px(ctx, FrameHeight(ctx)), available.y)};
}

void NodeIntInput::layout(const Context& ctx) {
    layoutChild(ctx, impl->input, frame());
}

struct NodeUIntInput::Impl {
    Config config;
    NodeTextInput input;
};

NodeUIntInput::NodeUIntInput() {
    this->impl = std::make_unique<Impl>();
    addChild(impl->input);
}

NodeUIntInput::~NodeUIntInput() = default;

void NodeUIntInput::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    impl->input.update({
        .id = impl->config.id + ":value",
        .value = jst::fmt::format("{}", impl->config.value),
        .unit = impl->config.unit,
        .submit = NodeTextInput::Submit::OnEnter,
        .selectAllOnFocus = true,
        .characterFilter = DecimalCharacter,
        .onChange = [this](const std::string& text) {
            const auto parsed = ParseUInt(text);
            if (parsed.has_value() && *parsed != impl->config.value && impl->config.onChange) {
                impl->config.onChange(*parsed);
            }
        },
    });
}

Extent2D<F32> NodeUIntInput::measure(const Context& ctx, Extent2D<F32> available) {
    return {FillLength(available.x), FitLength(Px(ctx, FrameHeight(ctx)), available.y)};
}

void NodeUIntInput::layout(const Context& ctx) {
    layoutChild(ctx, impl->input, frame());
}

struct NodeBoolInput::Impl {
    Config config;
    Box background;
    Label text;
    Label icon;
    TextMetrics metrics;
    Pressable press;
    Rect rect;
};

NodeBoolInput::NodeBoolInput() {
    this->impl = std::make_unique<Impl>();
    addChild(impl->background);
    addChild(impl->text);
    addChild(impl->icon);
}

NodeBoolInput::~NodeBoolInput() = default;

void NodeBoolInput::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
}

Extent2D<F32> NodeBoolInput::measure(const Context& ctx, Extent2D<F32> available) {
    return {FillLength(available.x), FitLength(Px(ctx, FrameHeight(ctx)), available.y)};
}

void NodeBoolInput::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    impl->rect = bounds;
    impl->metrics.setWindow(ctx.render);

    const F32 fontSize = Px(ctx, FontOf(ctx));
    const F32 pad = Px(ctx, FramePadX);
    const F32 iconSize = fontSize * 0.8f;
    const char* glyph = config.value ? ICON_FA_TOGGLE_ON : ICON_FA_TOGGLE_OFF;
    const F32 iconWidth = impl->metrics.measure(IconFont, glyph, iconSize);

    impl->background.update({
        .instances = {{.rect = Snap(bounds), .visible = !bounds.empty(), .backgroundColor = ctx.color("card")}},
        .clip = bounds,
        .cornerRadius = Px(ctx, FrameRounding),
    });
    impl->text.update({
        .instances = {{
            .rect = {bounds.x + pad, bounds.y, std::max(0.0f, bounds.width - pad), bounds.height},
            .str = config.value ? "Enabled" : "Disabled",
            .visible = !bounds.empty(),
            .color = ctx.color("text_primary"),
            .fontSize = fontSize,
            .alignment = {0, 1},
        }},
        .clip = bounds,
        .fontName = MonoFont,
        .maxCharacters = 16,
    });
    impl->icon.update({
        .instances = {{
            .rect = {bounds.right() - pad - iconWidth, bounds.y, iconWidth, bounds.height},
            .str = glyph,
            .visible = !bounds.empty(),
            .color = ctx.color(config.value ? "node_toggle_on" : "text_secondary"),
            .fontSize = iconSize,
            .alignment = {0, 1},
        }},
        .clip = bounds,
        .fontName = IconFont,
        .maxCharacters = 4,
    });

    layoutChild(ctx, impl->background, bounds);
    layoutChild(ctx, impl->text, bounds);
    layoutChild(ctx, impl->icon, bounds);
}

std::optional<SurfaceCursor> NodeBoolInput::cursorAt(const Extent2D<F32>& point) const {
    if (!Intersect(Intersect(impl->rect, frame()), clip()).contains(point.x, point.y)) {
        return std::nullopt;
    }
    return SurfaceCursor::Hand;
}

bool NodeBoolInput::mouseEvent(const MouseEvent& event) {
    if (impl->press.click(event, impl->rect)) {
        return true;
    }
    if (impl->press.release(event, impl->rect)) {
        if (impl->config.onChange) {
            impl->config.onChange(!impl->config.value);
        }
        return true;
    }
    return false;
}

struct NodeRangeInput::Impl {
    Config config;
    Box background;
    Box fill;
    Box knob;
    Label labels;
    NodeTextInput editor;
    TextMetrics metrics;
    Rect rect;
    bool dragging = false;
    bool editing = false;
    std::optional<F32> dragValue;

    F32 valueAt(F32 x) const {
        const F32 range = config.max - config.min;
        const F32 fraction = rect.width > 0.0f ? std::clamp((x - rect.x) / rect.width, 0.0f, 1.0f) : 0.0f;
        const F32 value = config.min + fraction * range;
        return config.integer ? std::round(value) : value;
    }

    void drag(NodeRangeInput& self, F32 x) {
        const F32 next = valueAt(x);
        const F32 current = dragValue.value_or(config.value);
        if (next != current) {
            dragValue = next;
            self.invalidate(Dirty::Layout);
            if (config.onChange) {
                config.onChange(next);
            }
        }
    }
};

NodeRangeInput::NodeRangeInput() {
    this->impl = std::make_unique<Impl>();
    addChild(impl->background);
    addChild(impl->fill);
    addChild(impl->knob);
    addChild(impl->labels);
    addChild(impl->editor);
    impl->editor.setVisible(false);
}

NodeRangeInput::~NodeRangeInput() = default;

void NodeRangeInput::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    if (!impl->dragging && impl->dragValue.has_value()) {
        if (*impl->dragValue != impl->config.value) {
            invalidate(Dirty::Layout);
        }
        impl->dragValue.reset();
    }
    impl->editor.update({
        .id = impl->config.id + ":edit",
        .value = FormatGeneral(impl->config.value),
        .submit = NodeTextInput::Submit::OnEnter,
        .selectAllOnFocus = true,
        .characterFilter = ScientificCharacter,
        .onChange = [this](const std::string& text) {
            const auto parsed = ParseFloat(text);
            if (parsed.has_value() && impl->config.onChange) {
                impl->config.onChange(static_cast<F32>(*parsed));
            }
        },
        .onDeactivate = [this]() {
            impl->editing = false;
            impl->editor.setVisible(false);
        },
    });
}

Extent2D<F32> NodeRangeInput::measure(const Context& ctx, Extent2D<F32> available) {
    return {FillLength(available.x), FitLength(Px(ctx, FrameHeight(ctx)), available.y)};
}

void NodeRangeInput::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    impl->rect = bounds;
    impl->metrics.setWindow(ctx.render);

    const bool visible = !impl->editing && !bounds.empty();
    const F32 value = impl->dragValue.value_or(config.value);
    const F32 range = config.max - config.min;
    const F32 fraction = std::clamp(range != 0.0f ? (value - config.min) / range : 0.0f, 0.0f, 1.0f);
    const F32 rounding = Px(ctx, FrameRounding);
    const F32 knobWidth = std::max(Px(ctx, 8.0f), rounding * 2.0f);
    const F32 knobX = std::clamp(bounds.x + bounds.width * fraction - knobWidth * 0.5f,
                                 bounds.x,
                                 std::max(bounds.x, bounds.right() - knobWidth));

    impl->background.update({
        .instances = {{.rect = Snap(bounds), .visible = visible, .backgroundColor = ctx.color("card")}},
        .clip = bounds,
        .cornerRadius = rounding,
    });
    impl->fill.update({
        .instances = {{
            .rect = Snap({bounds.x, bounds.y, knobX + knobWidth - bounds.x, bounds.height}),
            .visible = visible,
            .backgroundColor = ctx.color("frame_bg_hovered"),
        }},
        .clip = bounds,
        .cornerRadius = rounding,
    });
    impl->knob.update({
        .instances = {{
            .rect = Snap({knobX, bounds.y, knobWidth, bounds.height}),
            .visible = visible,
            .backgroundColor = ctx.color("frame_bg_active"),
        }},
        .clip = bounds,
        .cornerRadius = rounding,
    });

    const F32 fontSize = Px(ctx, FontOf(ctx));
    const F32 pad = Px(ctx, FramePadX);
    const F32 unitWidth = impl->metrics.measure(MonoFont, config.unit, fontSize);
    impl->labels.update({
        .instances = {
            {
                .rect = {bounds.x + pad, bounds.y, std::max(0.0f, bounds.width - pad), bounds.height},
                .str = jst::fmt::format("{:.{}f}", value, config.integer ? 0 : std::max<I32>(0, config.precision)),
                .visible = visible,
                .color = ctx.color("text_primary"),
                .fontSize = fontSize,
                .alignment = {0, 1},
            },
            {
                .rect = {bounds.right() - unitWidth - Px(ctx, UnitInset), bounds.y, unitWidth, bounds.height},
                .str = config.unit,
                .visible = visible && !config.unit.empty(),
                .color = ctx.color("text_secondary"),
                .fontSize = fontSize,
                .alignment = {0, 1},
            },
        },
        .clip = bounds,
        .fontName = MonoFont,
        .maxCharacters = 64,
    });

    layoutChild(ctx, impl->background, bounds);
    layoutChild(ctx, impl->fill, bounds);
    layoutChild(ctx, impl->knob, bounds);
    layoutChild(ctx, impl->labels, bounds);
    if (impl->editing) {
        layoutChild(ctx, impl->editor, bounds);
    }
}

std::optional<SurfaceCursor> NodeRangeInput::cursorAt(const Extent2D<F32>& point) const {
    if (impl->editing || !Intersect(Intersect(impl->rect, frame()), clip()).contains(point.x, point.y)) {
        return std::nullopt;
    }
    return SurfaceCursor::Hand;
}

bool NodeRangeInput::mouseEvent(const MouseEvent& event) {
    if (impl->editing) {
        return mouseEventChildren(event);
    }

    const bool inside = impl->rect.contains(event.position.x, event.position.y);
    switch (event.type) {
        case MouseEventType::Click:
            if (event.button != MouseButton::Left || !inside) {
                return false;
            }
            if (event.clicks >= 2) {
                impl->dragging = false;
                impl->dragValue.reset();
                impl->editing = true;
                impl->editor.setVisible(true);
                impl->editor.focusWhenPlaced(true);
                return true;
            }
            impl->dragging = true;
            impl->drag(*this, event.position.x);
            return true;
        case MouseEventType::Move:
            if (impl->dragging) {
                impl->drag(*this, event.position.x);
                return true;
            }
            return false;
        case MouseEventType::Cancel:
            impl->dragging = false;
            return false;
        case MouseEventType::Release:
            if (impl->dragging && event.button == MouseButton::Left) {
                impl->dragging = false;
                return true;
            }
            return false;
        default:
            return false;
    }
}

struct NodeCombo::Impl {
    Config config;
    Box frame;
    Polyline arrow;
    Label preview;
    Dropdown dropdown;
    Rect rect;
    Pointer pointer;
    bool hovered = false;
};

NodeCombo::NodeCombo() {
    this->impl = std::make_unique<Impl>();
    addChild(impl->frame);
    addChild(impl->arrow);
    addChild(impl->preview);
}

NodeCombo::~NodeCombo() = default;

void NodeCombo::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    const auto& current = impl->config;
    impl->dropdown.update({
        .id = current.id,
        .options = current.options,
        .selected = SelectedOption(current),
        .padding = FramePadX,
        .rowPadding = {FramePadX, FramePadY},
        .onSelect = OptionSelected(impl->config),
    });
}

Extent2D<F32> NodeCombo::measure(const Context& ctx, Extent2D<F32> available) {
    return {FillLength(available.x), FitLength(Px(ctx, FrameHeight(ctx)), available.y)};
}

void NodeCombo::layout(const Context& ctx) {
    const auto& config = impl->config;
    Rect bounds = frame();
    bounds.height = std::min(bounds.height, Px(ctx, FrameHeight(ctx)));
    const Rect clipRect = Intersect(bounds, clip());
    impl->rect = bounds;
    impl->hovered = impl->pointer.inside(clipRect);

    const F32 rounding = Px(ctx, FrameRounding);
    const F32 arrowSize = bounds.height;
    const F32 valueX2 = std::max(bounds.x, bounds.right() - arrowSize);
    const bool visible = !bounds.empty();
    const auto frameColor = ctx.color(impl->hovered ? "frame_bg_hovered" : "card");
    const auto buttonColor = (impl->dropdown.isOpen() || impl->hovered) ? ctx.color("button_hovered") : ctx.color("card");
    const bool hasValue = valueX2 > bounds.x;

    impl->frame.update({
        .instances = {
            {.rect = Snap({bounds.x, bounds.y, valueX2 - bounds.x, bounds.height}),
             .visible = visible && hasValue, .backgroundColor = frameColor,
             .corners = static_cast<U8>(Box::TopLeft | Box::BottomLeft)},
            {.rect = Snap({valueX2, bounds.y, bounds.right() - valueX2, bounds.height}),
             .visible = visible, .backgroundColor = buttonColor,
             .corners = hasValue ? static_cast<U8>(Box::TopRight | Box::BottomRight) : Box::AllCorners},
        },
        .clip = bounds,
        .cornerRadius = rounding,
    });

    const F32 fontSize = Px(ctx, FontOf(ctx));
    const F32 padY = Px(ctx, FramePadY);
    const F32 r = fontSize * 0.40f;
    const Extent2D<F32> center = {valueX2 + padY + fontSize * 0.5f, bounds.y + padY + fontSize * 0.5f};
    impl->arrow.update({
        .instances = {{
            .points = {
                {center.x + 0.0f, center.y + 0.75f * r},
                {center.x - 0.866f * r, center.y - 0.75f * r},
                {center.x + 0.866f * r, center.y - 0.75f * r},
            },
            .color = ctx.color("text_primary"),
            .filled = true,
            .visible = visible && valueX2 + arrowSize - Px(ctx, FramePadX) <= bounds.right(),
        }},
        .clip = bounds,
        .fringe = Fringe(ctx),
    });

    const F32 padX = Px(ctx, FramePadX);
    const Rect previewRect = {bounds.x + padX, bounds.y + padY, std::max(0.0f, valueX2 - bounds.x - padX), fontSize};
    impl->preview.update({
        .instances = {{
            .rect = previewRect,
            .str = config.value.empty() ? std::string("Select") : config.value,
            .visible = visible,
            .color = ctx.color("text_primary"),
            .fontSize = fontSize,
            .alignment = {0, 1},
        }},
        .clip = Intersect(bounds, {bounds.x, bounds.y, std::max(0.0f, valueX2 - bounds.x), bounds.height}),
        .fontName = MonoFont,
        .maxCharacters = 128,
    });

    layoutChild(ctx, impl->frame, bounds);
    layoutChild(ctx, impl->arrow, bounds);
    layoutChild(ctx, impl->preview, bounds);

    impl->dropdown.request(ctx, bounds);
}

bool NodeCombo::mouseEvent(const MouseEvent& event) {
    if (impl->pointer.track(event, Intersect(frame(), clip()), {impl->rect})) {
        invalidate(Dirty::Layout);
    }
    if (event.type == MouseEventType::Click && event.button == MouseButton::Left &&
        impl->rect.contains(event.position.x, event.position.y)) {
        if (impl->dropdown.isOpen()) {
            impl->dropdown.close();
        } else {
            impl->dropdown.open();
        }
        return true;
    }
    return false;
}

struct NodePathInput::Impl {
    Config config;
    Box background;
    NodeTextInput input;
    Label icon;
    TextMetrics metrics;
    Pressable press;
    Rect buttonRect;
    Rect inputRect;
    Pointer pointer;
};

NodePathInput::NodePathInput() {
    this->impl = std::make_unique<Impl>();
    addChild(impl->background);
    addChild(impl->input);
    addChild(impl->icon);
}

NodePathInput::~NodePathInput() = default;

void NodePathInput::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    impl->input.update({
        .id = impl->config.id + ":path",
        .value = impl->config.value,
        .hint = "Select file...",
        .submit = NodeTextInput::Submit::OnEnter,
        .onChange = impl->config.onChange,
    });
}

Extent2D<F32> NodePathInput::measure(const Context& ctx, Extent2D<F32> available) {
    return {FillLength(available.x), FitLength(Px(ctx, FrameHeight(ctx)), available.y)};
}

void NodePathInput::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const Rect clipRect = Intersect(bounds, clip());
    impl->metrics.setWindow(ctx.render);

    const F32 frameHeight = Px(ctx, FrameHeight(ctx));
    const F32 buttonWidth = frameHeight;
    const F32 connector = Px(ctx, ItemSpacing);
    const F32 inputWidth = std::max(0.0f, bounds.width - buttonWidth - connector);
    impl->inputRect = {bounds.x, bounds.y, inputWidth, frameHeight};
    impl->buttonRect = {bounds.x + inputWidth + connector, bounds.y, buttonWidth, frameHeight};

    impl->background.update({
        .instances = {
            {.rect = Snap({bounds.x, bounds.y, inputWidth + connector + buttonWidth, frameHeight}),
             .visible = !bounds.empty(), .backgroundColor = ctx.color("card")},
        },
        .clip = bounds,
        .cornerRadius = Px(ctx, FrameRounding),
    });

    const F32 iconSize = Px(ctx, FontOf(ctx)) * 0.7f;
    const F32 iconWidth = impl->metrics.measure(IconFont, ICON_FA_FOLDER_OPEN, iconSize);
    impl->icon.update({
        .instances = {{
            .rect = {impl->buttonRect.x + (impl->buttonRect.width - iconWidth) * 0.5f, impl->buttonRect.y,
                     iconWidth, impl->buttonRect.height},
            .str = ICON_FA_FOLDER_OPEN,
            .visible = !bounds.empty(),
            .color = ctx.color("text_primary"),
            .fontSize = iconSize,
            .alignment = {0, 1},
        }},
        .clip = bounds,
        .fontName = IconFont,
        .maxCharacters = 4,
    });

    layoutChild(ctx, impl->background, bounds);
    layoutChild(ctx, impl->input, impl->inputRect);
    layoutChild(ctx, impl->icon, bounds);

    const bool buttonHovered = impl->pointer.inside(Intersect(impl->buttonRect, clipRect));
    const bool inputHovered = impl->pointer.inside(Intersect(impl->inputRect, clipRect));
    impl->pointer.following = false;
    if (!config.value.empty() && (buttonHovered || inputHovered) && config.tooltip &&
        !impl->pointer.tooltip(ctx, bounds, config.tooltip())) {
        requestFrame();
    }
}

bool NodePathInput::mouseEvent(const MouseEvent& event) {
    const bool tooltip = impl->config.tooltip && !impl->config.value.empty();
    const bool moved = impl->pointer.track(event, Intersect(frame(), clip()), {impl->buttonRect, impl->inputRect});
    if (moved && tooltip) {
        invalidate(Dirty::Layout);
    }
    if (mouseEventChildren(event)) {
        return true;
    }
    if (impl->press.click(event, impl->buttonRect)) {
        return true;
    }
    if (impl->press.release(event, impl->buttonRect)) {
        if (impl->config.onBrowse) {
            impl->config.onBrowse();
        }
        return true;
    }
    return false;
}

struct NodeInputRow::Impl {
    Config config;
    Box background;
    NodeTextInput shape;
    NodeTextInput axes;
    std::vector<std::unique_ptr<NodeCombo>> combos;

    F32 combosWidth(const Context& ctx) const {
        F32 width = 0.0f;
        for (const auto& combo : config.combos) {
            width += Px(ctx, combo.width) + Px(ctx, ItemSpacing);
        }
        return width;
    }

    bool stackedFor(const Context& ctx, F32 available) const {
        const F32 minShape = Px(ctx, config.shapeMinWidth);
        const F32 minAxes = config.onAxesChange ? Px(ctx, config.axesMinWidth) : 0.0f;
        const F32 fieldsSpacing = config.onAxesChange ? Px(ctx, ItemSpacing) : 0.0f;
        return available < minShape + minAxes + combosWidth(ctx) + fieldsSpacing;
    }

    F32 rows() const {
        return static_cast<F32>(config.combos.size()) + (config.onAxesChange ? 2.0f : 1.0f);
    }
};

NodeInputRow::NodeInputRow() {
    this->impl = std::make_unique<Impl>();
    addChild(impl->background);
    addChild(impl->shape);
    addChild(impl->axes);
}

NodeInputRow::~NodeInputRow() = default;

void NodeInputRow::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    impl->shape.update({
        .id = impl->config.id + ":shape",
        .value = impl->config.shape,
        .hint = impl->config.shapeHint,
        .submit = NodeTextInput::Submit::OnCommit,
        .onChange = impl->config.onShapeChange,
    });
    impl->axes.update({
        .id = impl->config.id + ":axes",
        .value = impl->config.axes,
        .hint = impl->config.axesHint,
        .submit = NodeTextInput::Submit::OnCommit,
        .onChange = impl->config.onAxesChange,
    });
    const bool axesVisible = static_cast<bool>(impl->config.onAxesChange);
    if (impl->axes.visible() != axesVisible) {
        impl->axes.setVisible(axesVisible);
    }
    while (impl->combos.size() < impl->config.combos.size()) {
        impl->combos.push_back(std::make_unique<NodeCombo>());
        addChild(*impl->combos.back());
    }
    for (U64 i = 0; i < impl->combos.size(); ++i) {
        impl->combos[i]->setVisible(i < impl->config.combos.size());
    }
    for (U64 i = 0; i < impl->config.combos.size(); ++i) {
        const auto& combo = impl->config.combos[i];
        impl->combos[i]->update({
            .id = impl->config.id + ":combo" + std::to_string(i),
            .options = combo.options,
            .value = combo.value,
            .onChange = combo.onChange,
        });
    }
}

Extent2D<F32> NodeInputRow::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 frameHeight = Px(ctx, FrameHeight(ctx));
    if (!impl->stackedFor(ctx, available.x)) {
        return {FillLength(available.x), FitLength(frameHeight, available.y)};
    }
    const F32 rows = impl->rows();
    return {FillLength(available.x), FitLength(frameHeight * rows + Px(ctx, StackedGap) * (rows - 1.0f), available.y)};
}

void NodeInputRow::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const F32 frameHeight = Px(ctx, FrameHeight(ctx));
    const F32 spacing = Px(ctx, ItemSpacing);
    const F32 stackedSpacing = Px(ctx, StackedGap);

    const bool stacked = impl->stackedFor(ctx, bounds.width);

    const F32 rows = impl->rows();
    const F32 backgroundHeight = stacked ? frameHeight * rows + stackedSpacing * (rows - 1.0f) : frameHeight;
    impl->background.update({
        .instances = {{
            .rect = Snap({bounds.x, bounds.y, bounds.width, backgroundHeight}),
            .visible = !bounds.empty(),
            .backgroundColor = ctx.color("card"),
        }},
        .clip = bounds,
        .cornerRadius = Px(ctx, FrameRounding),
    });
    layoutChild(ctx, impl->background, bounds);

    const F32 minShape = Px(ctx, config.shapeMinWidth);
    const F32 axesWidth = stacked ? 0.0f : (config.onAxesChange ? Px(ctx, config.axesMinWidth) : 0.0f);
    const F32 shapeWidth = stacked
        ? bounds.width
        : std::max(minShape, bounds.width - impl->combosWidth(ctx) - axesWidth - (config.onAxesChange ? spacing : 0.0f));

    layoutChild(ctx, impl->shape, {bounds.x, bounds.y, shapeWidth, frameHeight});

    F32 y = bounds.y;
    if (config.onAxesChange) {
        if (stacked) {
            y += frameHeight + stackedSpacing;
            layoutChild(ctx, impl->axes, {bounds.x, y, bounds.width, frameHeight});
        } else {
            layoutChild(ctx, impl->axes, {bounds.x + shapeWidth + spacing, bounds.y, axesWidth, frameHeight});
        }
    }

    if (stacked) {
        for (U64 i = 0; i < config.combos.size(); ++i) {
            y += frameHeight + stackedSpacing;
            layoutChild(ctx, *impl->combos[i], {bounds.x, y, bounds.width, frameHeight});
        }
    } else {
        F32 x = bounds.x + shapeWidth + (config.onAxesChange ? axesWidth + spacing : 0.0f) + spacing;
        for (U64 i = 0; i < config.combos.size(); ++i) {
            const F32 width = ResolveLength(ctx, config.combos[i].width, 0.0f, std::numeric_limits<F32>::infinity());
            layoutChild(ctx, *impl->combos[i], {x, bounds.y, width, frameHeight});
            x += width + spacing;
        }
    }
}

}  // namespace Jetstream::Sakura
