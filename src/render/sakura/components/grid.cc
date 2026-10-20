#include <jetstream/render/sakura/components/grid.hh>

#include <jetstream/render/sakura/components/scroller.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/clip_layer.hh"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

struct Grid::Impl {
    Config config;
    ClipLayer clipLayer;
    Scroller scroll;
    F32 scrollY = 0.0f;
    std::vector<F32> rowHeights;

    U64 columns() const {
        return std::max<U64>(1, config.columns);
    }

    F32 cellWidth(const Context& ctx, F32 width) const {
        const F32 padX = Px(ctx, config.cellPadding.x);
        const F32 outer = config.outerPadding ? padX : 0.0f;
        const F32 n = static_cast<F32>(columns());
        return std::max(0.0f, (width - 2.0f * outer - (n - 1.0f) * 2.0f * padX) / n);
    }

    template<typename Measure>
    F32 measureRows(const Context& ctx, F32 width, const Measure& measure) {
        const F32 padY = Px(ctx, config.cellPadding.y);
        const F32 cell = cellWidth(ctx, width);
        const U64 n = columns();
        rowHeights.assign((config.items.size() + n - 1) / n, 0.0f);
        for (U64 index = 0; index < config.items.size(); ++index) {
            auto* item = config.items[index];
            if (!item) {
                continue;
            }
            const F32 height = measure(*item, cell);
            rowHeights[index / n] = std::max(rowHeights[index / n], height + 2.0f * padY);
        }
        F32 total = 0.0f;
        for (const F32 height : rowHeights) {
            total += height;
        }
        return total;
    }
};

Grid::Grid() {
    impl = std::make_unique<Impl>();
    addChild(impl->scroll);
    addChild(impl->clipLayer);
}

Grid::~Grid() = default;

void Grid::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    std::vector<Component*> items;
    for (auto* item : config.items) {
        if (item) {
            items.push_back(item);
        }
    }
    impl->clipLayer.mount(items);
    impl->config = std::move(config);
}

Extent2D<F32> Grid::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& config = impl->config;
    const F32 infinite = std::numeric_limits<F32>::infinity();
    F32 natural = 0.0f;
    if (config.size.x == 0.0f) {
        for (auto* item : config.items) {
            if (item) {
                natural = std::max(natural, measureChild(*item, ctx, {infinite, infinite}).x);
            }
        }
        const F32 padX = Px(ctx, config.cellPadding.x);
        const F32 n = static_cast<F32>(impl->columns());
        natural = n * natural + (n - 1.0f) * 2.0f * padX + (config.outerPadding ? 2.0f * padX : 0.0f);
    }
    const F32 width = ResolveLength(ctx, config.size.x, natural, available.x);
    const auto measureItem = [&](Component& item, F32 cell) {
        return measureChild(item, ctx, {cell, infinite}).y;
    };
    natural = config.size.y == 0.0f ? impl->measureRows(ctx, width, measureItem) : 0.0f;
    return {width, ResolveLength(ctx, config.size.y, natural, available.y)};
}

void Grid::layout(const Context& ctx) {
    const Rect bounds = frame();
    const auto& config = impl->config;
    const F32 padX = Px(ctx, config.cellPadding.x);
    const F32 padY = Px(ctx, config.cellPadding.y);
    const F32 outer = config.outerPadding ? padX : 0.0f;
    const auto measureItem = [&](Component& item, F32 cell) {
        return measureChild(item, ctx, {cell, std::numeric_limits<F32>::infinity()}).y;
    };
    F32 width = bounds.width;
    F32 contentHeight = impl->measureRows(ctx, width, measureItem);
    const F32 scrollbar = Px(ctx, kScrollbarSize);
    if (contentHeight > bounds.height + 0.5f) {
        width = std::max(0.0f, bounds.width - scrollbar);
        contentHeight = impl->measureRows(ctx, width, measureItem);
    }
    impl->scrollY = std::clamp(impl->scrollY, 0.0f, std::max(0.0f, contentHeight - bounds.height));

    impl->scroll.update({
        .contentHeight = contentHeight,
        .scrollY = impl->scrollY,
        .wheelStep = Px(ctx, kWheelStep),
        .thickness = scrollbar,
        .margin = 0.0f,
        .thumbInset = Px(ctx, kScrollbarGrabInset),
        .trackColorKey = "scrollbar_bg",
        .thumbColorKey = "scrollbar_grab",
        .onScrollY = [this](F32 next) {
            impl->scrollY = next;
            invalidate(Dirty::Layout);
        },
    });
    layoutChild(ctx, impl->scroll, bounds);
    layoutChild(ctx, impl->clipLayer, bounds);

    const U64 n = impl->columns();
    const F32 cell = impl->cellWidth(ctx, width);
    F32 y = bounds.y - impl->scrollY;
    for (U64 row = 0; row < impl->rowHeights.size(); ++row) {
        for (U64 column = 0; column < n; ++column) {
            const U64 index = row * n + column;
            if (index >= config.items.size() || !config.items[index]) {
                continue;
            }
            const F32 x = bounds.x + outer + static_cast<F32>(column) * (cell + 2.0f * padX);
            impl->clipLayer.place(ctx, *config.items[index],
                                  {x, y + padY, cell, std::max(0.0f, impl->rowHeights[row] - 2.0f * padY)});
        }
        y += impl->rowHeights[row];
    }
}

}  // namespace Jetstream::Sakura
