#include <jetstream/render/sakura/components/background_layer.hh>

#include <jetstream/render/sakura/components/box.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

struct BackgroundLayer::Impl {
    struct Particle {
        Extent2D<F32> position;
        F32 velocity = 0.0f;
        F32 radius = 0.0f;
        F32 alphaSeed = 0.0f;
        F32 phase = 0.0f;
    };

    Config config;
    Box fill;
    Box particles;
    std::vector<Particle> state;
    Rect area;
    std::mt19937 random{std::random_device{}()};
};

BackgroundLayer::BackgroundLayer() {
    impl = std::make_unique<Impl>();
    addChild(impl->fill);
    addChild(impl->particles);
}

BackgroundLayer::~BackgroundLayer() = default;

bool BackgroundLayer::covers(const Extent2D<F32>&) const {
    return false;
}

void BackgroundLayer::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    if (impl->config.particleCount != config.particleCount) {
        impl->state.clear();
    }
    impl->config = std::move(config);
}

void BackgroundLayer::layout(const Context& ctx) {
    const Rect bounds = frame();
    const auto& config = impl->config;
    const auto background = ctx.color(config.backgroundColorKey);
    impl->fill.update({.instances = {{.rect = bounds, .visible = !bounds.empty(), .backgroundColor = background}}});
    layoutChild(ctx, impl->fill, bounds);

    const F32 top = Px(ctx, config.topOffset);
    const Rect area = {bounds.x, bounds.y + top, bounds.width, std::max(0.0f, bounds.height - top)};
    if (!config.particles || config.particleCount == 0 || area.empty()) {
        impl->particles.update({.instances = {}});
        layoutChild(ctx, impl->particles, bounds);
        return;
    }

    requestFrame();
    const F32 unit = ctx.pixelRatio;
    std::uniform_real_distribution<F32> uniform(0.0f, 1.0f);
    if (impl->state.size() != config.particleCount || impl->area != area) {
        impl->area = area;
        impl->state.clear();
        for (U64 index = 0; index < config.particleCount; ++index) {
            impl->state.push_back({
                .position = {area.x + uniform(impl->random) * area.width, area.y + uniform(impl->random) * area.height},
                .velocity = 10.0f + uniform(impl->random) * 30.0f,
                .radius = 0.8f + uniform(impl->random) * 1.8f,
                .alphaSeed = uniform(impl->random),
                .phase = uniform(impl->random) * 6.28318f,
            });
        }
    }

    const auto base = ctx.color(config.particleColorKey);
    const F32 luminance = 0.2126f * background.r + 0.7152f * background.g + 0.0722f * background.b;
    const bool dark = luminance < 0.5f;
    const F32 alphaLo = dark ? 0.2f : 0.28f;
    const F32 alphaHi = dark ? 0.6f : 0.58f;
    const F32 time = static_cast<F32>(ctx.time);
    const F32 delta = ctx.deltaTime;

    std::vector<Box::Instance> instances;
    instances.reserve(impl->state.size());
    for (auto& particle : impl->state) {
        particle.position.x += particle.velocity * 0.15f * delta * unit;
        particle.position.y -= particle.velocity * delta * unit;
        const F32 wave = std::sin(time * 0.5f + particle.phase) * 2.0f * unit;
        const Extent2D<F32> center = {particle.position.x + wave, particle.position.y};
        if (center.y < area.y - 10.0f * unit) {
            particle.position.y = area.bottom() + 10.0f * unit;
            particle.position.x = area.x + uniform(impl->random) * area.width;
        }
        if (center.x > area.right() + 10.0f * unit) {
            particle.position.x = area.x - 10.0f * unit;
        }
        const F32 twinkle = std::sin(time * 2.0f + particle.phase) * 0.1f + 0.9f;
        auto color = base;
        color.a = (alphaLo + particle.alphaSeed * (alphaHi - alphaLo)) * twinkle;
        const F32 radius = particle.radius * unit;
        instances.push_back({
            .rect = {center.x - radius, center.y - radius, 2.0f * radius, 2.0f * radius},
            .visible = center.y >= area.y,
            .backgroundColor = color,
        });
    }
    impl->particles.update({
        .instances = std::move(instances),
        .clip = area,
        .cornerRadius = 64.0f,
        .capacity = std::max<U64>(config.particleCount, 1),
    });
    layoutChild(ctx, impl->particles, bounds);
}

}  // namespace Jetstream::Sakura
