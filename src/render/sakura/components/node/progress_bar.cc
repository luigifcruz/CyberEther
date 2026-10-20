#include <jetstream/render/sakura/components/node/progress_bar.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/node/content_stack.hh>

#include "../../detail/helpers.hh"
#include "../../detail/text_metrics.hh"
#include "../../metrics.hh"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

namespace Jetstream::Sakura {

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
    return {FillLength(available.x), FitLength(Px(ctx, NodeFontOf(ctx) + 10.0f), available.y)};
}

void NodeProgressBar::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const F32 fraction = std::clamp(config.value, 0.0f, 1.0f);
    const F32 rounding = Px(ctx, kFrameRounding);
    const F32 fontSize = Px(ctx, NodeFontOf(ctx));
    impl->metrics.setWindow(ctx.render);

    impl->frame.update({
        .instances = {{.rect = Snap(bounds), .visible = !bounds.empty(), .backgroundColor = ctx.color("card")}},
        .clip = bounds,
        .cornerRadius = rounding,
    });

    const F32 fillEnd = std::round(bounds.x + bounds.width * fraction);
    impl->fill.update({
        .instances = {{.rect = Snap(bounds), .visible = !bounds.empty() && fillEnd > bounds.x,
                       .backgroundColor = ctx.color(config.colorKey)}},
        .clip = Intersect(bounds, {bounds.x, bounds.y, std::max(0.0f, fillEnd - bounds.x), bounds.height}),
        .cornerRadius = rounding,
    });

    const F32 textWidth = impl->metrics.measure(Typography::MonoFont, config.overlay, fontSize);
    const F32 textX = std::round(std::clamp(fillEnd + Px(ctx, NodeContentStack::Spacing),
                                            bounds.x,
                                            std::max(bounds.x, bounds.right() - textWidth - Px(ctx, 8.0f))));
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
        .maxCharacters = 256,
    });

    layoutChild(ctx, impl->frame, bounds);
    layoutChild(ctx, impl->fill, bounds);
    layoutChild(ctx, impl->overlay, bounds);
}

}  // namespace Jetstream::Sakura
