#include <jetstream/render/sakura/components/node/bool_input.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/tools/icons_fa.hh>

#include "../../detail/helpers.hh"
#include "../../detail/text_metrics.hh"
#include "../../metrics.hh"
#include "pointer.hh"

#include <algorithm>
#include <utility>

namespace Jetstream::Sakura {

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
    return {FillLength(available.x), FitLength(Px(ctx, NodeFrameHeight(ctx)), available.y)};
}

void NodeBoolInput::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    impl->rect = bounds;
    impl->metrics.setWindow(ctx.render);

    const F32 fontSize = Px(ctx, NodeFontOf(ctx));
    const F32 pad = Px(ctx, kNodeFramePadX);
    const F32 iconSize = fontSize * 0.8f;
    const char* glyph = config.value ? ICON_FA_TOGGLE_ON : ICON_FA_TOGGLE_OFF;
    const F32 iconWidth = impl->metrics.measure(Typography::NodeIconFont, glyph, iconSize);

    impl->background.update({
        .instances = {{.rect = Snap(bounds), .visible = !bounds.empty(), .backgroundColor = ctx.color("card")}},
        .clip = bounds,
        .cornerRadius = Px(ctx, kFrameRounding),
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
        .fontName = Typography::MonoFont,
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
        .fontName = Typography::NodeIconFont,
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

}  // namespace Jetstream::Sakura
