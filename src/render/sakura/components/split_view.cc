#include <jetstream/render/sakura/components/split_view.hh>

#include <jetstream/render/sakura/components/box.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

struct SplitView::Impl {
    Config config;
    Box line;
};

SplitView::SplitView() {
    impl = std::make_unique<Impl>();
    setClipsChildren(true);
}

SplitView::~SplitView() = default;

void SplitView::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    const bool remount = config.left != impl->config.left || config.right != impl->config.right ||
                         !hasChild(impl->line) || (config.left && !hasChild(*config.left)) ||
                         (config.right && !hasChild(*config.right));
    if (remount) {
        std::vector<Component*> children = {&impl->line};
        if (config.left) {
            children.push_back(config.left);
        }
        if (config.right) {
            children.push_back(config.right);
        }
        mountChildren(children);
    }
    impl->config = std::move(config);
}

Extent2D<F32> SplitView::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& config = impl->config;
    const F32 width = FillLength(available.x);
    F32 natural = 0.0f;
    if (config.height == 0.0f) {
        const F32 infinite = std::numeric_limits<F32>::infinity();
        const F32 padX = Px(ctx, config.gutter);
        const F32 leftWidth = std::min(Px(ctx, config.leftWidth), width);
        const F32 rightWidth = std::max(0.0f, width - leftWidth - 2.0f * padX - Hairline(ctx));
        if (config.left) {
            natural = std::max(natural, measureChild(*config.left, ctx, {leftWidth, infinite}).y);
        }
        if (config.right) {
            natural = std::max(natural, measureChild(*config.right, ctx, {rightWidth, infinite}).y);
        }
        natural += 2.0f * Px(ctx, kCellPaddingY);
    }
    return {width, ResolveLength(ctx, config.height, natural, available.y)};
}

void SplitView::layout(const Context& ctx) {
    const Rect bounds = frame();
    const F32 padX = Px(ctx, impl->config.gutter);
    const F32 padY = Px(ctx, kCellPaddingY);
    const F32 leftWidth = std::min(Px(ctx, impl->config.leftWidth), bounds.width);
    const F32 line = Hairline(ctx);
    const F32 lineX = std::floor(bounds.x + leftWidth + padX);
    impl->line.update({
        .instances = {{.rect = {lineX, bounds.y, line, bounds.height},
                       .visible = !bounds.empty(),
                       .backgroundColor = ctx.color("table_border_light")}},
        .clip = bounds,
    });
    layoutChild(ctx, impl->line, bounds);
    const F32 innerHeight = std::max(0.0f, bounds.height - 2.0f * padY);
    if (impl->config.left) {
        layoutChild(ctx, *impl->config.left, {bounds.x, bounds.y + padY, leftWidth, innerHeight});
    }
    if (impl->config.right) {
        const F32 rightX = lineX + line + padX;
        layoutChild(ctx, *impl->config.right,
                    {rightX, bounds.y + padY, std::max(0.0f, bounds.right() - rightX), innerHeight});
    }
}

}  // namespace Jetstream::Sakura
