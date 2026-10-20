#include <jetstream/render/sakura/components/combo.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/dropdown.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/helpers.hh"
#include "../detail/options.hh"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>

namespace Jetstream::Sakura {

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

}  // namespace Jetstream::Sakura
