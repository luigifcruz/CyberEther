#include <jetstream/render/sakura/components/wave_line.hh>

#include <jetstream/render/sakura/components/polyline.hh>

#include "../context.hh"
#include "../detail/helpers.hh"
#include "../metrics.hh"

#include <algorithm>
#include <bit>
#include <cmath>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr U64 kWaveCapacity = 16384;
constexpr F32 kWaveAmplitude = 5.0f;
constexpr F32 kWaveThickness = 1.6f;
constexpr const char* kWaveColor = "text_disabled";

}  // namespace

struct WaveLine::Impl {
    Config config;
    Polyline line;
};

WaveLine::WaveLine() {
    impl = std::make_unique<Impl>();
    mountChildren({&impl->line});
}

WaveLine::~WaveLine() = default;

void WaveLine::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
}

Extent2D<F32> WaveLine::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 height = std::ceil(Px(ctx, 2.0f * kWaveAmplitude + kWaveThickness) + 2.0f * Fringe(ctx));
    return {FillLength(available.x), FitLength(height, available.y)};
}

void WaveLine::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const F32 period = Px(ctx, std::max(1.0f, config.period));
    const F32 amplitude = Px(ctx, kWaveAmplitude);
    const F32 center = bounds.y + bounds.height * 0.5f;
    const F32 width = bounds.width * std::clamp(config.span, 0.0f, 1.0f);
    const F32 left = bounds.x + std::floor((bounds.width - width) * 0.5f);
    const Rect visible = Intersect(bounds, clip());
    const U64 capacity = std::max(kWaveCapacity, std::bit_ceil(WaveRowCapacity(ctx, period)));
    Polyline::Instance wave{
        .points = WavePoints(left, center, width, period, amplitude, visible.x, visible.right()),
        .color = ctx.color(kWaveColor),
        .thickness = Px(ctx, kWaveThickness),
    };
    wave.points.resize(std::min<U64>(wave.points.size(), capacity / kWavePointVertices));
    impl->line.update({
        .instances = {std::move(wave)},
        .clip = bounds,
        .fringe = Fringe(ctx),
        .capacity = capacity,
    });
    layoutChild(ctx, impl->line, bounds);
}

}  // namespace Jetstream::Sakura
