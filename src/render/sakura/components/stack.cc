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

    std::vector<F32> mainSizes(Stack& self, const Context& ctx, Extent2D<F32> available) const {
        std::vector<F32> sizes(config.items.size(), 0.0f);
        const bool vertical = config.axis == Axis::Vertical;
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
    const auto sizes = impl->mainSizes(*this, ctx, available);
    const U64 count = impl->activeCount();
    F32 main = count > 1 ? Px(ctx, impl->config.spacing) * static_cast<F32>(count - 1) : 0.0f;
    F32 cross = 0.0f;
    for (U64 i = 0; i < impl->config.items.size(); ++i) {
        const auto& item = impl->config.items[i];
        if (!impl->active(item)) {
            continue;
        }
        main += sizes[i];
        if (!vertical) {
            cross = std::max(cross, measureChild(*item.child, ctx, {sizes[i], available.y}).y);
        } else if (!std::isfinite(available.x)) {
            cross = std::max(cross, measureChild(*item.child, ctx, available).x);
        }
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
    const U64 count = impl->activeCount();
    const F32 spacing = Px(ctx, impl->config.spacing);

    F32 used = count > 1 ? spacing * static_cast<F32>(count - 1) : 0.0f;
    F32 grow = 0.0f;
    for (U64 i = 0; i < impl->config.items.size(); ++i) {
        if (impl->active(impl->config.items[i])) {
            used += sizes[i];
            grow += impl->config.items[i].grow;
        }
    }
    const F32 extent = vertical ? bounds.height : bounds.width;
    F32 shrink = 0.0f;
    for (U64 i = 0; i < impl->config.items.size(); ++i) {
        if (impl->active(impl->config.items[i])) {
            shrink += impl->config.items[i].shrink;
        }
    }
    if (used > extent && shrink > 0.0f) {
        const F32 overflow = used - extent;
        for (U64 i = 0; i < impl->config.items.size(); ++i) {
            const auto& item = impl->config.items[i];
            if (impl->active(item) && item.shrink > 0.0f) {
                const F32 cut = std::min(sizes[i], overflow * item.shrink / shrink);
                sizes[i] -= cut;
                used -= cut;
            }
        }
    }
    const F32 remaining = std::max(0.0f, extent - used);
    if (grow > 0.0f) {
        for (U64 i = 0; i < impl->config.items.size(); ++i) {
            const auto& item = impl->config.items[i];
            if (impl->active(item) && item.grow > 0.0f) {
                sizes[i] += remaining * item.grow / grow;
            }
        }
    }

    F32 cursor = vertical ? bounds.y : bounds.x;
    for (U64 i = 0; i < impl->config.items.size(); ++i) {
        const auto& item = impl->config.items[i];
        if (!impl->active(item)) {
            continue;
        }
        Rect rect;
        if (vertical) {
            rect = {bounds.x, cursor, bounds.width, sizes[i]};
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
