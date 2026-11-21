#include <jetstream/render/sakura/components/node/input_row.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/node/combo.hh>
#include <jetstream/render/sakura/components/node/content_stack.hh>
#include <jetstream/render/sakura/components/node/text_input.hh>

#include "../../detail/helpers.hh"
#include "../../metrics.hh"

#include <algorithm>
#include <limits>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kStackedGap = 3.0f;

}  // namespace

struct NodeInputRow::Impl {
    Config config;
    Box background;
    NodeTextInput shape;
    NodeTextInput axes;
    std::vector<std::unique_ptr<NodeCombo>> combos;

    F32 combosWidth(const Context& ctx) const {
        F32 width = 0.0f;
        for (const auto& combo : config.combos) {
            width += Px(ctx, combo.width) + Px(ctx, NodeContentStack::Spacing);
        }
        return width;
    }

    bool stackedFor(const Context& ctx, F32 available) const {
        const F32 minShape = Px(ctx, config.shapeMinWidth);
        const F32 minAxes = config.onAxesChange ? Px(ctx, config.axesMinWidth) : 0.0f;
        const F32 fieldsSpacing = config.onAxesChange ? Px(ctx, NodeContentStack::Spacing) : 0.0f;
        return available < minShape + minAxes + combosWidth(ctx) + fieldsSpacing;
    }

    F32 rows() const {
        return static_cast<F32>(config.combos.size()) + (config.onAxesChange ? 2.0f : 1.0f);
    }
};

NodeInputRow::NodeInputRow() {
    this->impl = std::make_unique<Impl>();
    addChild(impl->background);
    addChild(impl->shape);
    addChild(impl->axes);
}

NodeInputRow::~NodeInputRow() = default;

void NodeInputRow::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    impl->shape.update({
        .id = impl->config.id + ":shape",
        .value = impl->config.shape,
        .hint = impl->config.shapeHint,
        .submit = NodeTextInput::Submit::OnCommit,
        .onChange = impl->config.onShapeChange,
    });
    impl->axes.update({
        .id = impl->config.id + ":axes",
        .value = impl->config.axes,
        .hint = impl->config.axesHint,
        .submit = NodeTextInput::Submit::OnCommit,
        .onChange = impl->config.onAxesChange,
    });
    const bool axesVisible = static_cast<bool>(impl->config.onAxesChange);
    if (impl->axes.visible() != axesVisible) {
        impl->axes.setVisible(axesVisible);
    }
    while (impl->combos.size() < impl->config.combos.size()) {
        impl->combos.push_back(std::make_unique<NodeCombo>());
        addChild(*impl->combos.back());
    }
    for (U64 i = 0; i < impl->combos.size(); ++i) {
        impl->combos[i]->setVisible(i < impl->config.combos.size());
    }
    for (U64 i = 0; i < impl->config.combos.size(); ++i) {
        const auto& combo = impl->config.combos[i];
        impl->combos[i]->update({
            .id = jst::fmt::format("{}:combo{}", impl->config.id, i),
            .options = combo.options,
            .value = combo.value,
            .onChange = combo.onChange,
        });
    }
}

Extent2D<F32> NodeInputRow::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 frameHeight = Px(ctx, NodeFrameHeight(ctx));
    if (!impl->stackedFor(ctx, available.x)) {
        return {FillLength(available.x), FitLength(frameHeight, available.y)};
    }
    const F32 rows = impl->rows();
    return {FillLength(available.x), FitLength(frameHeight * rows + Px(ctx, kStackedGap) * (rows - 1.0f), available.y)};
}

void NodeInputRow::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const F32 frameHeight = Px(ctx, NodeFrameHeight(ctx));
    const F32 spacing = Px(ctx, NodeContentStack::Spacing);
    const F32 stackedSpacing = Px(ctx, kStackedGap);

    const bool stacked = impl->stackedFor(ctx, bounds.width);

    const F32 rows = impl->rows();
    const F32 backgroundHeight = stacked ? frameHeight * rows + stackedSpacing * (rows - 1.0f) : frameHeight;
    impl->background.update({
        .instances = {{
            .rect = Snap({bounds.x, bounds.y, bounds.width, backgroundHeight}),
            .visible = !bounds.empty(),
            .backgroundColor = ctx.color("card"),
        }},
        .clip = bounds,
        .cornerRadius = Px(ctx, kFrameRounding),
    });
    layoutChild(ctx, impl->background, bounds);

    const F32 minShape = Px(ctx, config.shapeMinWidth);
    const F32 axesWidth = stacked ? 0.0f : (config.onAxesChange ? Px(ctx, config.axesMinWidth) : 0.0f);
    const F32 shapeWidth = stacked
        ? bounds.width
        : std::max(minShape, bounds.width - impl->combosWidth(ctx) - axesWidth - (config.onAxesChange ? spacing : 0.0f));

    layoutChild(ctx, impl->shape, {bounds.x, bounds.y, shapeWidth, frameHeight});

    F32 y = bounds.y;
    if (config.onAxesChange) {
        if (stacked) {
            y += frameHeight + stackedSpacing;
            layoutChild(ctx, impl->axes, {bounds.x, y, bounds.width, frameHeight});
        } else {
            layoutChild(ctx, impl->axes, {bounds.x + shapeWidth + spacing, bounds.y, axesWidth, frameHeight});
        }
    }

    if (stacked) {
        for (U64 i = 0; i < config.combos.size(); ++i) {
            y += frameHeight + stackedSpacing;
            layoutChild(ctx, *impl->combos[i], {bounds.x, y, bounds.width, frameHeight});
        }
    } else {
        F32 x = bounds.x + shapeWidth + (config.onAxesChange ? axesWidth + spacing : 0.0f) + spacing;
        for (U64 i = 0; i < config.combos.size(); ++i) {
            const F32 width = ResolveLength(ctx, config.combos[i].width, 0.0f, std::numeric_limits<F32>::infinity());
            layoutChild(ctx, *impl->combos[i], {x, bounds.y, width, frameHeight});
            x += width + spacing;
        }
    }
}

}  // namespace Jetstream::Sakura
