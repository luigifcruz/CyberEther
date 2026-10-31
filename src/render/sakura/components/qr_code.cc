#include <jetstream/render/sakura/components/qr_code.hh>

#include <jetstream/render/sakura/components/box.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

struct QrCode::Impl {
    Config config;
    Box modules;

    static constexpr int kBorder = 2;

    F32 side(const Context& ctx, F32 limit) const {
        if (config.width <= 0 || config.data.empty() || config.heightRatio <= 0.0f) {
            return 0.0f;
        }
        const F32 modules = static_cast<F32>(config.width + 2 * kBorder);
        const F32 target = std::min(Px(ctx, config.heightRatio * LogicalViewportHeight(ctx)), limit);
        return modules * std::max(1.0f, std::floor(target / modules));
    }
};

QrCode::QrCode() {
    impl = std::make_unique<Impl>();
    addChild(impl->modules);
}

QrCode::~QrCode() = default;

void QrCode::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
}

Extent2D<F32> QrCode::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 side = impl->side(ctx, std::min(available.x, available.y));
    return FitExtent({std::isfinite(available.x) ? std::max(available.x, side) : side, side}, available);
}

void QrCode::layout(const Context& ctx) {
    const Rect bounds = frame();
    const auto& config = impl->config;
    const F32 side = impl->side(ctx, std::min(bounds.width, bounds.height));
    const F32 module = side > 0.0f ? side / static_cast<F32>(config.width + 2 * Impl::kBorder) : 0.0f;
    const F32 x0 = std::floor(bounds.x + std::max(0.0f, (bounds.width - side) * 0.5f));
    const F32 y0 = std::floor(bounds.y);

    std::vector<Box::Instance> instances;
    if (side > 0.0f) {
        instances.push_back({.rect = {x0, y0, side, side}, .backgroundColor = {1.0f, 1.0f, 1.0f, 1.0f}});
        for (int y = 0; y < config.width; ++y) {
            int x = 0;
            while (x < config.width) {
                const auto at = [&](int column) {
                    const U64 index = static_cast<U64>(y * config.width + column);
                    return index < config.data.size() && config.data[index] != 0;
                };
                if (!at(x)) {
                    ++x;
                    continue;
                }
                int end = x;
                while (end < config.width && at(end)) {
                    ++end;
                }
                instances.push_back({
                    .rect = {x0 + static_cast<F32>(x + Impl::kBorder) * module,
                             y0 + static_cast<F32>(y + Impl::kBorder) * module,
                             static_cast<F32>(end - x) * module, module},
                    .backgroundColor = {0.0f, 0.0f, 0.0f, 1.0f},
                });
                x = end;
            }
        }
    }
    impl->modules.update({.instances = std::move(instances), .clip = bounds});
    layoutChild(ctx, impl->modules, bounds);
}

}  // namespace Jetstream::Sakura
