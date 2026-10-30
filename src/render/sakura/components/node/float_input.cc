#include <jetstream/render/sakura/components/node/float_input.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/node/text_input.hh>

#include "../../detail/helpers.hh"
#include "../../metrics.hh"
#include "parse.hh"
#include "pointer.hh"

#include <algorithm>
#include <utility>

namespace Jetstream::Sakura {

namespace {

std::string FormatFloat(F32 value, I32 precision) {
    return jst::fmt::format("{:.{}f}", value, std::clamp(precision, 0, 12));
}

}  // namespace

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
    const F32 height = impl->stepped() ? NodeFrameHeight(ctx) + NodeFontOf(ctx) + 4.0f : NodeFrameHeight(ctx);
    return {FillLength(available.x), FitLength(Px(ctx, height), available.y)};
}

void NodeFloatInput::layout(const Context& ctx) {
    const Rect bounds = frame();
    const Rect clipRect = Intersect(bounds, clip());
    const F32 fontSize = Px(ctx, NodeFontOf(ctx));

    if (!impl->stepped()) {
        layoutChild(ctx, impl->input, bounds);
        return;
    }

    const F32 inputHeight = Px(ctx, NodeFrameHeight(ctx));
    const F32 buttonHeight = Px(ctx, NodeFontOf(ctx) + 4.0f);
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
        .cornerRadius = Px(ctx, kFrameRounding),
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
        .fontName = Typography::MonoFont,
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

}  // namespace Jetstream::Sakura
