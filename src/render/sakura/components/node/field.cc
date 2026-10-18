#include <jetstream/render/sakura/components/node/field.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>

#include "common.hh"

#include <algorithm>
#include <cmath>
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

struct NodeFieldGrid::Impl {
    Config config;

    struct Placement {
        Component* child = nullptr;
        Rect rect;
    };

    std::vector<Placement> arrange(NodeFieldGrid& self, const Context& ctx, F32 width, F32& height) {
        std::vector<Placement> placements;
        height = 0.0f;
        std::vector<NodeFieldGrid::Item> items;
        for (const auto& item : config.items) {
            if (!item.child || item.child->visible()) {
                items.push_back(item);
            }
        }
        if (items.empty()) {
            return placements;
        }

        const F32 gap = Px(ctx, NodeField::Gap);
        const F32 minColumnWidth = Px(ctx, FieldColumnMinWidth);
        const U64 columns = std::min<U64>(
            std::max<U64>(1, items.size()),
            (width > 0.0f && minColumnWidth > 0.0f)
                ? std::max<U64>(1, static_cast<U64>((width + gap) / (minColumnWidth + gap)))
                : 1);

        F32 y = 0.0f;
        bool first = true;
        const auto measure = [&](Component* child, F32 w) {
            if (!child || !child->visible()) {
                return 0.0f;
            }
            return self.measureChild(*child, ctx, {w, std::numeric_limits<F32>::infinity()}).y;
        };

        for (U64 i = 0; i < items.size();) {
            if (!first) {
                y += gap;
            }
            first = false;

            if (columns == 1 || items[i].fullWidth) {
                const F32 h = measure(items[i].child, width);
                placements.push_back({items[i].child, {0.0f, y, width, h}});
                y += h;
                ++i;
                continue;
            }

            const U64 groupStart = i;
            while (i < items.size() && !items[i].fullWidth) {
                ++i;
            }
            const U64 groupEnd = i;
            const U64 groupColumns = std::min(columns, groupEnd - groupStart);
            const F32 columnWidth = (width - static_cast<F32>(groupColumns - 1) * gap) / static_cast<F32>(groupColumns);

            F32 rowY = y;
            F32 rowHeight = 0.0f;
            std::vector<U64> rowItems;
            const auto flushRow = [&]() {
                for (const U64 index : rowItems) {
                    placements[index].rect.height = rowHeight;
                }
                rowItems.clear();
            };
            for (U64 j = groupStart; j < groupEnd; ++j) {
                const U64 column = (j - groupStart) % groupColumns;
                if (column == 0 && j != groupStart) {
                    flushRow();
                    rowY += rowHeight + gap;
                    rowHeight = 0.0f;
                }
                const F32 x = static_cast<F32>(column) * (columnWidth + gap);
                const F32 h = measure(items[j].child, columnWidth);
                rowItems.push_back(placements.size());
                placements.push_back({items[j].child, {x, rowY, columnWidth, h}});
                rowHeight = std::max(rowHeight, h);
            }
            flushRow();
            y = rowY + rowHeight;
        }

        height = y;
        return placements;
    }
};

NodeFieldGrid::NodeFieldGrid() {
    this->impl = std::make_unique<Impl>();
}

NodeFieldGrid::~NodeFieldGrid() = default;

void NodeFieldGrid::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    std::vector<Component*> active;
    active.reserve(config.items.size());
    for (const auto& item : config.items) {
        active.push_back(item.child);
    }
    mountChildren(active);
    impl->config = std::move(config);
}

Extent2D<F32> NodeFieldGrid::measure(const Context& ctx, Extent2D<F32> available) {
    F32 height = 0.0f;
    impl->arrange(*this, ctx, available.x, height);
    return {FillLength(available.x), FitLength(height, available.y)};
}

void NodeFieldGrid::layout(const Context& ctx) {
    const Rect bounds = frame();
    F32 height = 0.0f;
    const auto placements = impl->arrange(*this, ctx, bounds.width, height);
    for (const auto& placement : placements) {
        if (!placement.child || !placement.child->visible()) {
            continue;
        }
        const Rect rect = {
            bounds.x + placement.rect.x,
            bounds.y + placement.rect.y,
            placement.rect.width,
            placement.rect.height,
        };
        layoutChild(ctx, *placement.child, rect);
    }
}

struct NodeContentStack::Impl {
    Config config;

    static F32 FloorToF32(const F64 value) {
        F32 result = static_cast<F32>(std::min(value, static_cast<F64>(std::numeric_limits<F32>::max())));
        if (static_cast<F64>(result) > value) {
            result = std::nextafter(result, 0.0f);
        }
        return std::max(0.0f, result);
    }

    bool active(const Item& item) const {
        return item.child && item.child->visible();
    }

    F32 gapBefore(const Context& ctx, const Item& item) const {
        return Px(ctx, item.gap.value_or(ItemSpacing));
    }

    struct Arrangement {
        std::vector<F32> heights;
        std::vector<std::optional<F32>> allocations;
        F32 fixed = 0.0f;
        F32 flexibleMinimum = 0.0f;
        F32 total = 0.0f;
    };

    Arrangement arrange(NodeContentStack& self, const Context& ctx, F32 width, std::optional<F32> height) {
        Arrangement result;
        const auto& items = config.items;
        result.heights.assign(items.size(), 0.0f);
        result.allocations.assign(items.size(), std::nullopt);

        F64 minimumFlexible = 0.0;
        F64 basisFlexible = 0.0;
        F64 totalGrow = 0.0;
        U64 flexibleCount = 0;
        bool first = true;
        for (U64 i = 0; i < items.size(); ++i) {
            const auto& item = items[i];
            if (!active(item)) {
                continue;
            }
            if (!first) {
                result.fixed += gapBefore(ctx, item);
            }
            first = false;
            result.fixed += Px(ctx, item.trailingGap);
            if (item.flex.has_value() && height.has_value()) {
                const auto& flex = *item.flex;
                const F32 minimum = Px(ctx, flex.minimum);
                minimumFlexible += minimum;
                basisFlexible += Px(ctx, std::max(flex.minimum, flex.basis.value_or(flex.minimum)));
                totalGrow += flex.grow;
                ++flexibleCount;
                continue;
            }
            if (item.flex.has_value()) {
                result.flexibleMinimum += Px(ctx, item.flex->minimum);
            }
            result.heights[i] = self.measureChild(*item.child, ctx, {width, std::numeric_limits<F32>::infinity()}).y;
            if (item.flex.has_value()) {
                result.heights[i] = std::max(result.heights[i], Px(ctx, item.flex->minimum));
            }
            result.fixed += result.heights[i];
        }
        result.flexibleMinimum += static_cast<F32>(minimumFlexible);

        if (height.has_value() && flexibleCount > 0) {
            const F64 available = std::max(0.0f, *height - result.fixed);
            const F64 growHeight = std::max(0.0, available - basisFlexible);
            const F64 growUnit = totalGrow > 0.0 ? growHeight / totalGrow : 0.0;
            const F64 minimumScale = minimumFlexible > 0.0 ? std::min(1.0, available / minimumFlexible) : 1.0;
            const F64 basisScale = basisFlexible > minimumFlexible
                ? std::clamp((available - minimumFlexible) / (basisFlexible - minimumFlexible), 0.0, 1.0)
                : 1.0;
            U64 lastFlexible = 0;
            for (U64 i = 0; i < items.size(); ++i) {
                if (active(items[i]) && items[i].flex.has_value()) {
                    lastFlexible = i;
                }
            }
            const bool fillAvailable = available <= basisFlexible || totalGrow > 0.0;
            F64 remaining = available;
            for (U64 i = 0; i < items.size(); ++i) {
                const auto& item = items[i];
                if (!active(item) || !item.flex.has_value()) {
                    continue;
                }
                const auto& flex = *item.flex;
                const F64 minimum = Px(ctx, flex.minimum);
                const F64 basis = Px(ctx, std::max(flex.minimum, flex.basis.value_or(flex.minimum)));
                F64 h = minimum * minimumScale;
                if (available >= minimumFlexible) {
                    h = minimum + (basis - minimum) * basisScale;
                }
                if (available >= basisFlexible) {
                    h = basis + growUnit * static_cast<F64>(flex.grow);
                }
                const F64 bounded = fillAvailable && i == lastFlexible ? remaining : std::min(remaining, h);
                const F32 allocated = std::max(FloorToF32(bounded), static_cast<F32>(minimum));
                result.heights[i] = allocated;
                result.allocations[i] = allocated;
                remaining = std::max(0.0, remaining - static_cast<F64>(allocated));
            }
        }

        result.total = 0.0f;
        first = true;
        for (U64 i = 0; i < items.size(); ++i) {
            if (!active(items[i])) {
                continue;
            }
            if (!first) {
                result.total += gapBefore(ctx, items[i]);
            }
            first = false;
            result.total += result.heights[i] + Px(ctx, items[i].trailingGap);
        }
        return result;
    }
};

NodeContentStack::NodeContentStack() {
    this->impl = std::make_unique<Impl>();
}

NodeContentStack::~NodeContentStack() = default;

void NodeContentStack::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    std::vector<Component*> active;
    active.reserve(config.items.size());
    for (const auto& item : config.items) {
        active.push_back(item.child);
    }
    mountChildren(active);
    impl->config = std::move(config);
}

bool NodeContentStack::empty() const {
    for (const auto& item : impl->config.items) {
        if (impl->active(item)) {
            return false;
        }
    }
    return true;
}

F32 NodeContentStack::leadingGap() const {
    for (const auto& item : impl->config.items) {
        if (impl->active(item)) {
            return item.gap.value_or(NodeStyle::ItemSpacing);
        }
    }
    return NodeStyle::ItemSpacing;
}

bool NodeContentStack::hasFlexible() const {
    for (const auto& item : impl->config.items) {
        if (impl->active(item) && item.flex.has_value()) {
            return true;
        }
    }
    return false;
}

NodeContentStack::Metrics NodeContentStack::metrics(const Context& ctx, F32 width) {
    const auto constrained = impl->arrange(*this, ctx, width, 0.0f);
    F32 flexibleBasis = 0.0f;
    for (const auto& item : impl->config.items) {
        if (impl->active(item) && item.flex.has_value()) {
            const auto& flex = *item.flex;
            flexibleBasis += Px(ctx, std::max(flex.minimum, flex.basis.value_or(flex.minimum)));
        }
    }
    return {
        .natural = hasFlexible() ? impl->arrange(*this, ctx, width, std::nullopt).total : constrained.total,
        .fixed = constrained.fixed,
        .minimum = constrained.fixed + constrained.flexibleMinimum,
        .flexibleBasis = flexibleBasis,
    };
}

Extent2D<F32> NodeContentStack::measure(const Context& ctx, Extent2D<F32> available) {
    const auto arrangement = impl->arrange(*this, ctx, available.x, std::nullopt);
    return {FillLength(available.x), FitLength(arrangement.total, available.y)};
}

void NodeContentStack::layout(const Context& ctx) {
    const Rect bounds = frame();
    const auto arrangement = impl->arrange(*this, ctx, bounds.width, hasFlexible() ? std::optional<F32>(bounds.height)
                                                                                   : std::nullopt);
    F32 y = bounds.y;
    bool first = true;
    const auto& items = impl->config.items;
    for (U64 i = 0; i < items.size(); ++i) {
        const auto& item = items[i];
        if (!impl->active(item)) {
            continue;
        }
        if (!first) {
            y += impl->gapBefore(ctx, item);
        }
        first = false;
        layoutChild(ctx, *item.child, {bounds.x, y, bounds.width, arrangement.heights[i]});
        y += arrangement.heights[i] + Px(ctx, item.trailingGap);
    }
}

}  // namespace Jetstream::Sakura
