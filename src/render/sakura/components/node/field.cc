#include <jetstream/render/sakura/components/node/field.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>

#include "common.hh"

#include <algorithm>
#include <limits>
#include <utility>

namespace Jetstream::Sakura {

using namespace NodeStyle;

struct NodeField::Impl {
    Config config;
    Box card;
    Label label;
    TextMetrics metrics;
    Pointer pointer;
    Rect hoverRect;

    F32 childTop(const Context& ctx) const {
        if (!config.title || config.label.empty()) {
            return 0.0f;
        }
        if (config.background) {
            return Px(ctx, CardLabelOffsetY + LabelFont);
        }
        return Px(ctx, LabelFont + 4.0f);
    }
};

NodeField::NodeField() {
    this->impl = std::make_unique<Impl>();
}

NodeField::~NodeField() = default;

void NodeField::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    auto* content = impl->config.content;
    const bool titled = impl->config.title && !impl->config.label.empty();
    std::vector<Component*> next;
    if (titled && impl->config.background) {
        next.push_back(&impl->card);
    }
    if (titled) {
        next.push_back(&impl->label);
    }
    for (auto* mounted : children()) {
        if (mounted != &impl->card && mounted != &impl->label) {
            next.push_back(mounted);
        }
    }
    if (content && std::find(next.begin(), next.end(), content) == next.end()) {
        next.push_back(content);
    }
    setChildren(next);
    for (auto* mounted : children()) {
        if (mounted != &impl->card && mounted != &impl->label) {
            mounted->setVisible(mounted == content);
        }
    }
}

Extent2D<F32> NodeField::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 top = impl->childTop(ctx);
    F32 childHeight = 0.0f;
    if (impl->config.content && impl->config.content->visible()) {
        childHeight = measureChild(*impl->config.content, ctx, {available.x, std::numeric_limits<F32>::infinity()}).y;
    }
    F32 height = top + childHeight;
    if (impl->config.title && !impl->config.label.empty() && impl->config.background) {
        height = std::max(height, Px(ctx, CardLabelHeight(ctx)));
    }
    return {FillLength(available.x), FitLength(height, available.y)};
}

void NodeField::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const Rect clipRect = Intersect(bounds, clip());
    const bool titled = config.title && !config.label.empty();

    impl->metrics.setWindow(ctx.render);

    if (titled && config.background) {
        impl->card.update({
            .instances = {{
                .rect = Snap({bounds.x, bounds.y, bounds.width, Px(ctx, CardLabelHeight(ctx))}),
                .visible = !bounds.empty(),
                .backgroundColor = ctx.color("card"),
            }},
            .clip = bounds,
            .cornerRadius = Px(ctx, FrameRounding),
        });
        layoutChild(ctx, impl->card, bounds);
    }

    const F32 labelY = bounds.y + (config.background ? Px(ctx, CardLabelOffsetY) : 0.0f);
    const F32 labelX = bounds.x + Px(ctx, CardLabelOffsetX);
    const F32 labelSize = Px(ctx, LabelFont);
    const F32 labelWidth = impl->metrics.measure(MonoFont, config.label, labelSize);
    const Rect labelRect = {labelX, labelY, std::max(0.0f, bounds.right() - labelX), labelSize};

    if (titled) {
        impl->label.update({
            .instances = {{
                .rect = labelRect,
                .str = config.label,
                .visible = !bounds.empty(),
                .color = ctx.color("text_secondary"),
                .fontSize = labelSize,
                .alignment = {0, 1},
            }},
            .clip = Intersect(bounds, labelRect),
            .fontName = MonoFont,
            .maxCharacters = 128,
        });
        layoutChild(ctx, impl->label, bounds);
    }

    const F32 top = impl->childTop(ctx);
    if (impl->config.content && impl->config.content->visible()) {
        Context childCtx = ctx;
        if (titled && config.background) {
            childCtx.backdropTopColorKey = "card";
        }
        layoutChild(childCtx, *impl->config.content, {bounds.x, bounds.y + top, bounds.width, std::max(0.0f, bounds.height - top)});
    }

    impl->hoverRect = Intersect({labelX, labelY, std::min(labelWidth, labelRect.width), labelSize}, clipRect);
    impl->pointer.following = false;
    if (titled && config.tooltip && impl->pointer.inside(impl->hoverRect) &&
        !impl->pointer.tooltip(ctx, impl->hoverRect, config.tooltip())) {
        requestFrame();
    }
}

bool NodeField::mouseEvent(const MouseEvent& event) {
    const bool titled = impl->config.title && !impl->config.label.empty();
    const bool moved = impl->pointer.track(event, impl->hoverRect, {impl->hoverRect});
    if (moved && titled && impl->config.tooltip) {
        invalidate(Dirty::Layout);
    }
    return mouseEventChildren(event);
}

}  // namespace Jetstream::Sakura
