#include <jetstream/render/sakura/components/progress_bar.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/stack.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/helpers.hh"
#include "../detail/text_metrics.hh"

#include <algorithm>
#include <cmath>
#include <utility>

namespace Jetstream::Sakura {

struct ProgressBar::Impl {
    Config config;
    Box background;
    Box fill;
    Label overlay;
    TextMetrics metrics;
};

ProgressBar::ProgressBar() {
    impl = std::make_unique<Impl>();
    addChild(impl->background);
    addChild(impl->fill);
    addChild(impl->overlay);
}

ProgressBar::~ProgressBar() = default;

void ProgressBar::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
}

Extent2D<F32> ProgressBar::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& size = impl->config.size;
    return {ResolveLength(ctx, size.x, Px(ctx, kNaturalControlWidth), available.x),
            ResolveLength(ctx, size.y, FrameHeightPx(ctx), available.y)};
}

void ProgressBar::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const F32 rounding = Px(ctx, kFrameRounding);
    const F32 fraction = std::clamp(config.value, 0.0f, 1.0f);
    const F32 fillEnd = fraction <= 0.0f ? bounds.x
                      : fraction >= 1.0f ? bounds.right()
                      : std::clamp(std::round(bounds.x + bounds.width * fraction), bounds.x, bounds.right());
    impl->background.update({
        .instances = {{.rect = bounds, .visible = !bounds.empty(),
                       .backgroundColor = ctx.color(config.backgroundColorKey.empty() ? "card"
                                                                                      : config.backgroundColorKey)}},
        .clip = bounds,
        .cornerRadius = rounding,
    });
    impl->fill.update({
        .instances = {{.rect = bounds, .visible = !bounds.empty() && fillEnd > bounds.x,
                       .backgroundColor = ctx.color(config.colorKey)}},
        .clip = Intersect(bounds, {bounds.x, bounds.y, std::max(0.0f, fillEnd - bounds.x), bounds.height}),
        .cornerRadius = rounding,
    });
    impl->metrics.setWindow(ctx.render);
    const F32 fontSize = FontPx(ctx);
    const F32 textWidth = impl->metrics.measure(Typography::MonoFont, config.overlay, fontSize);
    const F32 textX = std::round(std::clamp(fillEnd + Px(ctx, Stack::Spacing), bounds.x,
                                            std::max(bounds.x, bounds.right() - textWidth - Px(ctx, kInnerSpacing))));
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
        .maxCharacters = 128,
    });
    layoutChild(ctx, impl->background, bounds);
    layoutChild(ctx, impl->fill, bounds);
    layoutChild(ctx, impl->overlay, bounds);
}

}  // namespace Jetstream::Sakura
