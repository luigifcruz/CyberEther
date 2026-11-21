#include <jetstream/render/sakura/components/combo.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/card.hh>
#include <jetstream/render/sakura/components/dropdown.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/render/tools/icons_lucide.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/helpers.hh"
#include "../detail/options.hh"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kComboText = 14.0f;
constexpr F32 kComboPadX = 14.0f;
constexpr F32 kComboChevron = 13.0f;

}  // namespace

struct Combo::Impl {
    Config config;
    Card card;
    Box hover;
    Label chevron;
    Label preview;
    Dropdown dropdown;
    bool hovered = false;
};

Combo::Combo() {
    impl = std::make_unique<Impl>();
    setClipsChildren(false);
    addChild(impl->card);
    addChild(impl->hover);
    addChild(impl->chevron);
    addChild(impl->preview);
}

Combo::~Combo() = default;

void Combo::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    const auto& current = impl->config;
    if (current.disabled) {
        impl->hovered = false;
        if (impl->dropdown.isOpen()) {
            impl->dropdown.close();
        }
    }
    impl->dropdown.update({
        .id = current.id,
        .options = current.options,
        .selected = SelectedOption(current),
        .padding = 8.0f,
        .rowPadding = {6.0f, 4.0f},
        .fontName = Typography::DisplayFont,
        .onSelect = OptionSelected(impl->config),
    });
}

Extent2D<F32> Combo::measure(const Context& ctx, Extent2D<F32> available) {
    return {ResolveLength(ctx, impl->config.width, Px(ctx, kNaturalControlWidth), available.x),
            FitLength(Px(ctx, kPillHeight), available.y)};
}

void Combo::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const F32 alpha = config.disabled ? kDisabledAlpha : 1.0f;
    const bool active = impl->dropdown.isOpen() || (impl->hovered && !config.disabled);
    const bool visible = !bounds.empty();
    const F32 radius = bounds.height * 0.5f;

    impl->card.update({
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
    impl->hover.update({
        .instances = {{.rect = bounds, .visible = visible && active, .backgroundColor = ctx.color(kPillHover)}},
        .cornerRadius = radius,
        .capacity = 1,
    });

    const F32 fontSize = Px(ctx, kComboText);
    const F32 padX = Px(ctx, kComboPadX);
    const F32 chevronSize = Px(ctx, kComboChevron);
    const F32 chevronX = bounds.right() - padX - chevronSize;
    const F32 valueX2 = std::max(bounds.x + padX, chevronX - Px(ctx, kPillIconGap));

    impl->chevron.update({
        .instances = {{
            .rect = {chevronX, bounds.y, chevronSize, bounds.height},
            .str = ICON_LC_CHEVRON_DOWN,
            .visible = visible && chevronX > bounds.x + padX,
            .color = WithAlpha(ctx.color(active ? "text_primary" : "text_secondary"), alpha),
            .fontSize = chevronSize,
            .alignment = {0, 1},
        }},
        .clip = bounds,
        .fontName = Typography::IconFont,
        .maxCharacters = 4,
        .capacity = 1,
    });

    impl->preview.update({
        .instances = {{
            .rect = {bounds.x + padX, bounds.y, std::max(0.0f, valueX2 - bounds.x - padX), bounds.height},
            .str = config.value.empty() ? std::string("Select") : config.value,
            .visible = visible,
            .color = WithAlpha(ctx.color(config.value.empty() ? "text_secondary" : "text_primary"), alpha),
            .fontSize = fontSize,
            .alignment = {0, 1},
        }},
        .clip = Rect{bounds.x, bounds.y, std::max(0.0f, valueX2 - bounds.x), bounds.height},
        .fontName = Typography::DisplayFont,
        .maxCharacters = 128,
    });

    layoutChild(ctx, impl->card, bounds);
    layoutChild(ctx, impl->hover, bounds);
    layoutChild(ctx, impl->chevron, bounds);
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
