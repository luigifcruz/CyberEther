#include <jetstream/render/sakura/components/node/range_input.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/node/text_input.hh>

#include "../../detail/helpers.hh"
#include "../../detail/text_metrics.hh"
#include "../../metrics.hh"
#include "parse.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace Jetstream::Sakura {

namespace {

std::string FormatGeneral(F32 value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*g", std::numeric_limits<F32>::max_digits10, static_cast<double>(value));
    return buffer;
}

}  // namespace

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
    if (config.id != impl->config.id && (impl->dragging || impl->editing)) {
        impl->dragging = false;
        impl->dragValue.reset();
        impl->editing = false;
        impl->editor.setVisible(false);
        cancelMouse(impl->editor);
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
    return {FillLength(available.x), FitLength(Px(ctx, NodeFrameHeight(ctx)), available.y)};
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
    const F32 rounding = Px(ctx, kFrameRounding);
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

    const F32 fontSize = Px(ctx, NodeFontOf(ctx));
    const F32 pad = Px(ctx, kNodeFramePadX);
    const F32 unitWidth = impl->metrics.measure(Typography::MonoFont, config.unit, fontSize);
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
                .rect = {bounds.right() - unitWidth - Px(ctx, kNodeUnitInset), bounds.y, unitWidth, bounds.height},
                .str = config.unit,
                .visible = visible && !config.unit.empty(),
                .color = ctx.color("text_secondary"),
                .fontSize = fontSize,
                .alignment = {0, 1},
            },
        },
        .clip = bounds,
        .fontName = Typography::MonoFont,
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

}  // namespace Jetstream::Sakura
