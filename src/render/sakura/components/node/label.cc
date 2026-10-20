#include <jetstream/render/sakura/components/node/label.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>

#include "../../detail/helpers.hh"
#include "../../detail/text_metrics.hh"
#include "../../metrics.hh"

#include <algorithm>
#include <memory>
#include <utility>

namespace Jetstream::Sakura {

struct NodeLabel::Impl {
    Config config;
    Box frame;
    Label text;
    TextMetrics metrics;
};

NodeLabel::NodeLabel() {
    this->impl = std::make_unique<Impl>();
    addChild(impl->text);
}

NodeLabel::~NodeLabel() = default;

void NodeLabel::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    if (impl->config.boxed) {
        setChildren({&impl->frame, &impl->text});
    } else {
        setChildren({&impl->text});
    }
}

Extent2D<F32> NodeLabel::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 line = Px(ctx, NodeFontOf(ctx));
    return {FillLength(available.x), FitLength(impl->config.boxed ? line + Px(ctx, 16.0f) : line, available.y)};
}

void NodeLabel::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const F32 fontSize = Px(ctx, NodeFontOf(ctx));
    const F32 padding = config.boxed ? std::min(Px(ctx, 8.0f), bounds.width * 0.5f) : 0.0f;
    const F32 paddingY = config.boxed ? Px(ctx, 8.0f) : 0.0f;
    const Rect textRect = {bounds.x + padding, bounds.y + paddingY, std::max(0.0f, bounds.width - 2.0f * padding), fontSize};

    if (config.boxed) {
        impl->frame.update({
            .instances = {{
                .rect = Snap(bounds),
                .visible = !bounds.empty(),
                .backgroundColor = ctx.color("card"),
            }},
            .clip = bounds,
            .cornerRadius = Px(ctx, 6.0f),
            .borderWidth = Hairline(ctx),
            .borderColor = ctx.color("table_border_strong"),
        });
    }

    impl->metrics.setWindow(ctx.render);
    std::string line = config.str.substr(0, config.str.find('\n'));
    const bool overflows = config.align != Align::Left &&
                           impl->metrics.measure(Typography::MonoFont, line, fontSize) > textRect.width;
    const I32 alignX = overflows                     ? 0
                       : config.align == Align::Center ? 1
                       : config.align == Align::Right  ? 2
                                                       : 0;
    impl->text.update({
        .instances = {{
            .rect = textRect,
            .str = std::move(line),
            .visible = !bounds.empty(),
            .color = ctx.color(config.colorKey),
            .fontSize = fontSize,
            .alignment = {alignX, 1},
        }},
        .clip = Intersect(bounds, textRect),
        .fontName = Typography::MonoFont,
        .maxCharacters = 256,
    });

    if (config.boxed) {
        layoutChild(ctx, impl->frame, bounds);
    }
    layoutChild(ctx, impl->text, bounds);
}

}  // namespace Jetstream::Sakura
