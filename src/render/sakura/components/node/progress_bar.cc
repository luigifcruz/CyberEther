#include <jetstream/render/sakura/components/node/progress_bar.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/node/content_stack.hh>

#include "../../detail/helpers.hh"
#include "../../detail/text_metrics.hh"
#include "../../metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F64 kColorFadeSeconds = 0.25;

}  // namespace

struct NodeProgressBar::Impl {
    Config config;
    Box frame;
    Box fill;
    Label overlay;
    TextMetrics metrics;
    std::string fadeKey;
    ColorRGBA<F32> fadeFrom;
    F64 fadeStart = -std::numeric_limits<F64>::infinity();

    ColorRGBA<F32> blend(const ColorRGBA<F32>& target, F64 now) const {
        const F32 t = static_cast<F32>(std::clamp((now - fadeStart) / kColorFadeSeconds, 0.0, 1.0));
        return t >= 1.0f ? target : MixColor(fadeFrom, target, Smoothstep(t));
    }

    ColorRGBA<F32> fillColor(const Context& ctx, bool& fading) {
        if (fadeKey != config.colorKey) {
            if (!fadeKey.empty()) {
                fadeFrom = blend(ctx.color(fadeKey), ctx.time);
                fadeStart = ctx.time;
            }
            fadeKey = config.colorKey;
        }
        fading = ctx.time - fadeStart < kColorFadeSeconds;
        return blend(ctx.color(config.colorKey), ctx.time);
    }
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

    const F32 fillEnd = fraction <= 0.0f ? bounds.x
                      : fraction >= 1.0f ? bounds.right()
                      : std::clamp(std::round(bounds.x + bounds.width * fraction), bounds.x, bounds.right());
    bool fading = false;
    const auto fillColor = impl->fillColor(ctx, fading);
    if (fading) {
        requestFrame();
    }
    impl->fill.update({
        .instances = {{.rect = Snap(bounds), .visible = !bounds.empty() && fillEnd > bounds.x,
                       .backgroundColor = fillColor}},
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
