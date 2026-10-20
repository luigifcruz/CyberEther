#include <jetstream/render/sakura/components/node/progress_bar.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>

#include "common.hh"
#include "../../metrics.hh"

#include <algorithm>
#include <memory>
#include <utility>

namespace Jetstream::Sakura {

using namespace NodeStyle;

struct NodeProgressBar::Impl {
    Config config;
    Box frame;
    Box fill;
    Label overlay;
    TextMetrics metrics;
};

NodeProgressBar::NodeProgressBar() {
    this->impl = std::make_unique<Impl>();
    addChild(impl->frame);
    addChild(impl->fill);
    addChild(impl->overlay);
}

NodeProgressBar::~NodeProgressBar() = default;

void NodeProgressBar::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
}

Extent2D<F32> NodeProgressBar::measure(const Context& ctx, Extent2D<F32> available) {
    return {FillLength(available.x), FitLength(Px(ctx, FontOf(ctx) + 10.0f), available.y)};
}

void NodeProgressBar::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const F32 fraction = std::clamp(config.value, 0.0f, 1.0f);
    const F32 rounding = Px(ctx, FrameRounding);
    const F32 fontSize = Px(ctx, FontOf(ctx));
    impl->metrics.setWindow(ctx.render);

    impl->frame.update({
        .instances = {{.rect = Snap(bounds), .visible = !bounds.empty(), .backgroundColor = ctx.color("card")}},
        .clip = bounds,
        .cornerRadius = rounding,
    });

    const F32 fillEnd = bounds.x + bounds.width * fraction;
    impl->fill.update({
        .instances = {{.rect = Snap(bounds), .visible = !bounds.empty() && fraction > 0.0f,
                       .backgroundColor = ctx.color(config.colorKey)}},
        .clip = Intersect(bounds, {bounds.x, bounds.y, std::max(0.0f, fillEnd - bounds.x), bounds.height}),
        .cornerRadius = rounding,
    });

    const F32 textWidth = impl->metrics.measure(MonoFont, config.overlay, fontSize);
    const F32 textX = std::clamp(fillEnd + Px(ctx, ItemSpacing),
                                 bounds.x,
                                 std::max(bounds.x, bounds.right() - textWidth - Px(ctx, 8.0f)));
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
        .fontName = MonoFont,
        .maxCharacters = 256,
    });

    layoutChild(ctx, impl->frame, bounds);
    layoutChild(ctx, impl->fill, bounds);
    layoutChild(ctx, impl->overlay, bounds);
}

}  // namespace Jetstream::Sakura
