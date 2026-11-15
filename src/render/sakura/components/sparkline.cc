#include <jetstream/render/sakura/components/sparkline.hh>

#include <jetstream/render/sakura/components/box.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/helpers.hh"

#include <algorithm>
#include <cmath>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kNaturalHeight = 28.0f;
constexpr F32 kGapWidth = 3.0f;
constexpr F32 kHeadroom = 2.0f;

}  // namespace

struct Sparkline::Impl {
    Config config;
    Box bars;
    Box line;
};

Sparkline::Sparkline() {
    impl = std::make_unique<Impl>();
    addChild(impl->line);
    addChild(impl->bars);
}

Sparkline::~Sparkline() = default;

void Sparkline::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
}

Extent2D<F32> Sparkline::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& size = impl->config.size;
    return {ResolveLength(ctx, size.x, Px(ctx, kNaturalControlWidth), available.x),
            ResolveLength(ctx, size.y, Px(ctx, kNaturalHeight), available.y)};
}

void Sparkline::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const U64 capacity = std::max<U64>(1, config.capacity);
    const U64 slots = std::clamp<U64>(static_cast<U64>(bounds.width), 1, capacity);
    const F32 step = bounds.width / static_cast<F32>(slots);
    const F32 gap = step >= Px(ctx, kGapWidth) ? Hairline(ctx) : 0.0f;
    const U64 shown = std::min<U64>(slots, config.values.size());
    const U64 first = config.values.size() - shown;

    F32 ceiling = config.limit * kHeadroom;
    if (ceiling <= 0.0f) {
        for (U64 i = first; i < config.values.size(); ++i) {
            ceiling = std::max(ceiling, config.values[i]);
        }
    }
    ceiling = std::max(ceiling, 1.0e-6f);

    const auto color = ctx.color(config.colorKey);
    const auto alertColor = ctx.color(config.alertColorKey);
    std::vector<Box::Instance> instances;
    instances.reserve(shown);
    const auto edge = [&](U64 slot) {
        return std::round(bounds.x + static_cast<F32>(slot) * step);
    };
    for (U64 i = 0; i < shown; ++i) {
        const U64 slot = slots - shown + i;
        const F32 left = edge(slot);
        const F32 width = std::max(1.0f, edge(slot + 1) - left - gap);
        const F32 value = config.values[first + i];
        const F32 fraction = std::isfinite(value) ? std::clamp(value / ceiling, 0.0f, 1.0f) : 0.0f;
        const F32 height = std::max(1.0f, std::round(bounds.height * fraction));
        instances.push_back({
            .rect = {left, bounds.bottom() - height, width, height},
            .visible = !bounds.empty(),
            .backgroundColor = config.alert > 0.0f && value > config.alert ? alertColor : color,
        });
    }
    impl->bars.update({
        .instances = std::move(instances),
        .clip = bounds,
        .capacity = capacity,
    });

    const bool ruled = config.limit > 0.0f && !bounds.empty();
    const F32 rule = Hairline(ctx);
    const F32 level = std::round(bounds.bottom() - bounds.height * std::clamp(config.limit / ceiling, 0.0f, 1.0f));
    impl->line.update({
        .instances = {{.rect = {bounds.x, level, bounds.width, rule}, .visible = ruled,
                       .backgroundColor = ctx.color(config.limitColorKey)}},
        .clip = bounds,
        .capacity = 1,
    });

    layoutChild(ctx, impl->line, bounds);
    layoutChild(ctx, impl->bars, bounds);
}

}  // namespace Jetstream::Sakura
