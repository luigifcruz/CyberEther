#include <jetstream/render/sakura/components/spinner.hh>

#include <jetstream/render/sakura/components/polyline.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kDiameter = 16.0f;
constexpr U64 kSegments = 48;
constexpr F32 kSweep = 0.75f;
constexpr F32 kTrackAlpha = 0.2f;

std::vector<Extent2D<F32>> Arc(Extent2D<F32> center, F32 radius, F32 start, F32 sweep, U64 segments) {
    std::vector<Extent2D<F32>> points;
    points.reserve(segments + 1);
    for (U64 index = 0; index <= segments; ++index) {
        const F32 angle = start + sweep * static_cast<F32>(index) / static_cast<F32>(segments);
        points.push_back({center.x + radius * std::cos(angle), center.y + radius * std::sin(angle)});
    }
    return points;
}

}  // namespace

struct Spinner::Impl {
    Config config;
    Polyline arc;
};

Spinner::Spinner() {
    impl = std::make_unique<Impl>();
    addChild(impl->arc);
}

Spinner::~Spinner() = default;

void Spinner::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
}

Extent2D<F32> Spinner::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 diameter = impl->config.diameter;
    const F32 natural = Px(ctx, kDiameter);
    const F32 side = std::min(ResolveLength(ctx, diameter, natural, available.x),
                              ResolveLength(ctx, diameter, natural, available.y));
    return {side, side};
}

void Spinner::layout(const Context& ctx) {
    auto& s = *impl;
    const Rect bounds = frame();
    const F32 thickness = Px(ctx, s.config.thickness);
    const F32 radius = std::max(0.0f, std::min(bounds.width, bounds.height) * 0.5f - thickness * 0.5f);
    const Extent2D<F32> center = {bounds.x + bounds.width * 0.5f, bounds.y + bounds.height * 0.5f};
    const F32 tau = 2.0f * std::numbers::pi_v<F32>;
    const F64 turns = s.config.period > 0.0f ? ctx.time / static_cast<F64>(s.config.period) : 0.0;
    const F32 start = static_cast<F32>(turns - std::floor(turns)) * tau;
    const auto color = ctx.color(s.config.colorKey);
    const bool drawn = radius > 0.0f;

    s.arc.update({
        .instances = {
            {.points = Polyline::Circle(center, radius, kSegments), .color = WithAlpha(color, kTrackAlpha),
             .thickness = thickness, .closed = true, .visible = drawn},
            {.points = Arc(center, radius, start, kSweep * tau, kSegments), .color = color, .thickness = thickness,
             .visible = drawn},
        },
        .clip = clip(),
        .fringe = Fringe(ctx),
        .capacity = 8 * (kSegments + 1),
    });
    layoutChild(ctx, s.arc, bounds);
    if (drawn && s.config.period > 0.0f) {
        requestFrame();
    }
}

}  // namespace Jetstream::Sakura
