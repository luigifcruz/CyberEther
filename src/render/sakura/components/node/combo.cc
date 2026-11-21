#include <jetstream/render/sakura/components/node/combo.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/dropdown.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/polyline.hh>

#include "../../detail/helpers.hh"
#include "../../detail/options.hh"
#include "../../metrics.hh"
#include "pointer.hh"

#include <algorithm>
#include <utility>

namespace Jetstream::Sakura {

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

const NodeCombo::Config& NodeCombo::config() const {
    return impl->config;
}

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
        .padding = kNodeFramePadX,
        .rowPadding = {kNodeFramePadX, kNodeFramePadY},
        .onSelect = OptionSelected(impl->config),
    });
}

Extent2D<F32> NodeCombo::measure(const Context& ctx, Extent2D<F32> available) {
    return {FillLength(available.x), FitLength(Px(ctx, NodeFrameHeight(ctx)), available.y)};
}

void NodeCombo::layout(const Context& ctx) {
    const auto& config = impl->config;
    Rect bounds = frame();
    bounds.height = std::min(bounds.height, Px(ctx, NodeFrameHeight(ctx)));
    const Rect clipRect = Intersect(bounds, clip());
    impl->rect = bounds;
    impl->hovered = impl->pointer.inside(clipRect);

    const F32 rounding = Px(ctx, kFrameRounding);
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

    const F32 fontSize = Px(ctx, NodeFontOf(ctx));
    const F32 padY = Px(ctx, kNodeFramePadY);
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
            .visible = visible && valueX2 + arrowSize - Px(ctx, kNodeFramePadX) <= bounds.right(),
        }},
        .clip = bounds,
        .fringe = Fringe(ctx),
    });

    const F32 padX = Px(ctx, kNodeFramePadX);
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
        .fontName = Typography::MonoFont,
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

}  // namespace Jetstream::Sakura
