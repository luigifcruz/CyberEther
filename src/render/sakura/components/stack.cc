#include <jetstream/render/sakura/components/stack.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

struct Stack::Impl {
    Config config;

    bool active(const Item& item) const {
        return item.child && item.child->visible();
    }

    bool columned() const {
        if (config.axis != Axis::Vertical || !config.widthOf) {
            return false;
        }
        return std::any_of(config.items.begin(), config.items.end(), [&](const Item& item) {
            return item.child == config.widthOf && active(item);
        });
    }

    F32 columnWidth(Stack& self, const Context& ctx, F32 available) const {
        if (!columned()) {
            return available;
        }
        const F32 width = self.measureChild(*config.widthOf, ctx, {available, std::numeric_limits<F32>::infinity()}).x;
        return std::isfinite(available) ? std::min(width, available) : width;
    }

    std::vector<F32> mainSizes(Stack& self, const Context& ctx, Extent2D<F32> available) const {
        std::vector<F32> sizes(config.items.size(), 0.0f);
        const bool vertical = config.axis == Axis::Vertical;
        available.x = columnWidth(self, ctx, available.x);
        for (U64 i = 0; i < config.items.size(); ++i) {
            const auto& item = config.items[i];
            if (!active(item)) {
                continue;
            }
            if (item.size > 0.0f) {
                sizes[i] = Px(ctx, item.size);
                continue;
            }
            const Extent2D<F32> bounds = vertical
                ? Extent2D<F32>{available.x, std::numeric_limits<F32>::infinity()}
                : Extent2D<F32>{std::numeric_limits<F32>::infinity(), available.y};
            const auto measured = self.measureChild(*item.child, ctx, bounds);
            sizes[i] = vertical ? measured.y : measured.x;
        }
        return sizes;
    }

    U64 activeCount() const {
        U64 count = 0;
        for (const auto& item : config.items) {
            count += active(item) ? 1 : 0;
        }
        return count;
    }

    F32 used(const Context& ctx, const std::vector<F32>& sizes) const {
        const U64 count = activeCount();
        F32 total = count > 1 ? Px(ctx, config.spacing) * static_cast<F32>(count - 1) : 0.0f;
        for (U64 i = 0; i < config.items.size(); ++i) {
            total += active(config.items[i]) ? sizes[i] : 0.0f;
        }
        return total;
    }

    void shrink(const Context& ctx, std::vector<F32>& sizes, F32 extent) const {
        if (!std::isfinite(extent)) {
            return;
        }
        const auto shrinkable = [&](U64 i) {
            const auto& item = config.items[i];
            return active(item) && item.shrink > 0.0f && sizes[i] > 0.0f;
        };
        F32 overflow = used(ctx, sizes) - extent;
        for (U64 round = 0; round < config.items.size() && overflow > 0.0f; ++round) {
            F32 weight = 0.0f;
            for (U64 i = 0; i < config.items.size(); ++i) {
                weight += shrinkable(i) ? config.items[i].shrink : 0.0f;
            }
            if (weight <= 0.0f) {
                return;
            }
            F32 removed = 0.0f;
            for (U64 i = 0; i < config.items.size(); ++i) {
                if (shrinkable(i)) {
                    const F32 cut = std::min(sizes[i], overflow * config.items[i].shrink / weight);
                    sizes[i] -= cut;
                    removed += cut;
                }
            }
            overflow -= removed;
        }
    }
};

Stack::Stack() {
    impl = std::make_unique<Impl>();
}

Stack::~Stack() = default;

void Stack::update(Config config) {
    const bool mounted = std::all_of(config.items.begin(), config.items.end(), [this](const Item& item) {
        return !item.child || hasChild(*item.child);
    });
    if (mounted && config == impl->config) {
        impl->config = std::move(config);
        return;
    }
    invalidate(Dirty::Layout);
    std::vector<Component*> children;
    children.reserve(config.items.size());
    for (const auto& item : config.items) {
        if (item.child) {
            children.push_back(item.child);
        }
    }
    mountChildren(children);
    impl->config = std::move(config);
}

Extent2D<F32> Stack::measure(const Context& ctx, Extent2D<F32> available) {
    const bool vertical = impl->config.axis == Axis::Vertical;
    auto sizes = impl->mainSizes(*this, ctx, available);
    impl->shrink(ctx, sizes, vertical ? available.y : available.x);
    const F32 main = impl->used(ctx, sizes);
    F32 cross = 0.0f;
    for (U64 i = 0; i < impl->config.items.size(); ++i) {
        const auto& item = impl->config.items[i];
        if (!impl->active(item)) {
            continue;
        }
        if (!vertical) {
            cross = std::max(cross, measureChild(*item.child, ctx, {sizes[i], available.y}).y);
        } else if (!std::isfinite(available.x) && !impl->columned()) {
            cross = std::max(cross, measureChild(*item.child, ctx, available).x);
        }
    }
    if (vertical && impl->columned()) {
        cross = impl->columnWidth(*this, ctx, available.x);
    }
    if (vertical) {
        return {FillLength(available.x, cross), FitLength(main, available.y)};
    }
    return FitExtent({main, cross}, available);
}

void Stack::layout(const Context& ctx) {
    const Rect bounds = frame();
    const bool vertical = impl->config.axis == Axis::Vertical;
    auto sizes = impl->mainSizes(*this, ctx, {bounds.width, bounds.height});
    const F32 spacing = Px(ctx, impl->config.spacing);
    const F32 extent = vertical ? bounds.height : bounds.width;
    impl->shrink(ctx, sizes, extent);

    F32 grow = 0.0f;
    for (const auto& item : impl->config.items) {
        grow += impl->active(item) ? item.grow : 0.0f;
    }
    const F32 remaining = std::max(0.0f, extent - impl->used(ctx, sizes));
    if (grow > 0.0f) {
        for (U64 i = 0; i < impl->config.items.size(); ++i) {
            const auto& item = impl->config.items[i];
            if (impl->active(item) && item.grow > 0.0f) {
                sizes[i] += remaining * item.grow / grow;
            }
        }
    }

    const F32 column = impl->columnWidth(*this, ctx, bounds.width);
    F32 cursor = vertical ? bounds.y : bounds.x;
    for (U64 i = 0; i < impl->config.items.size(); ++i) {
        const auto& item = impl->config.items[i];
        if (!impl->active(item)) {
            continue;
        }
        Rect rect;
        if (vertical) {
            rect = {bounds.x + std::floor((bounds.width - column) * 0.5f), cursor, column, sizes[i]};
        } else {
            F32 height = bounds.height;
            F32 y = bounds.y;
            if (impl->config.centerCross) {
                height = std::min(bounds.height, measureChild(*item.child, ctx, {sizes[i], bounds.height}).y);
                y = bounds.y + std::floor((bounds.height - height) * 0.5f);
            }
            rect = {cursor, y, sizes[i], height};
        }
        layoutChild(ctx, *item.child, rect);
        cursor += sizes[i] + spacing;
    }
}

}  // namespace Jetstream::Sakura
