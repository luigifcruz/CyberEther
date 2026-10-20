#include <jetstream/render/sakura/components/node/content_stack.hh>

#include "common.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace Jetstream::Sakura {

using namespace NodeStyle;

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
