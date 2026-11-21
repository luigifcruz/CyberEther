#include <jetstream/render/sakura/components/reveal.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr F64 kRevealDuration = 0.25;

}  // namespace

struct Reveal::Impl {
    Config config;
    bool stale = true;
    std::optional<F64> synced;
    std::optional<F64> seen;
    std::optional<F64> shownAt;
    std::optional<F64> hiddenAt;
    std::optional<F64> wake;
    F32 progress = 0.0f;

    bool showing() const {
        return config.shown || progress > 0.0f;
    }

    void present(Reveal& self, Component& child, bool visible) {
        if (!visible && child.visible() && self.hasChild(child)) {
            self.cancelMouse(child);
            self.releaseFocusWithin(child);
        }
        child.setVisible(visible);
    }

    void advance(F64 time) {
        if (!stale && synced == time) {
            return;
        }
        stale = false;
        synced = time;
        const F64 elapsed = seen.has_value() ? std::max(0.0, time - *seen) : 0.0;
        if (config.shown) {
            if (!shownAt.has_value()) {
                shownAt = time;
            }
            hiddenAt.reset();
        } else if (shownAt.has_value() && !hiddenAt.has_value()) {
            hiddenAt = time;
        }
        const F64 until = hiddenAt.has_value()
            ? std::max(*shownAt + config.minimumDuration, *hiddenAt + config.linger)
            : 0.0;
        const bool holding = hiddenAt.has_value() && time < until;
        const bool target = config.shown || holding;
        if (!config.shown && !holding) {
            shownAt.reset();
            hiddenAt.reset();
        }
        const F32 step = static_cast<F32>(elapsed / kRevealDuration);
        progress = std::clamp(target ? progress + step : progress - step, 0.0f, 1.0f);
        const bool settled = progress == (target ? 1.0f : 0.0f);
        seen = settled ? std::nullopt : std::optional<F64>(time);
        wake.reset();
        if (!settled) {
            wake = -std::numeric_limits<F64>::infinity();
        } else if (holding) {
            wake = until;
        }
    }
};

Reveal::Reveal() {
    impl = std::make_unique<Impl>();
}

Reveal::~Reveal() = default;

void Reveal::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->stale = impl->stale || config.shown != impl->config.shown ||
                  config.minimumDuration != impl->config.minimumDuration || config.linger != impl->config.linger;
    std::vector<Component*> children;
    if (config.child) {
        impl->present(*this, *config.child, config.shown || impl->progress > 0.0f);
        children.push_back(config.child);
    }
    setChildren(children);
    impl->config = std::move(config);
}

Extent2D<F32> Reveal::measure(const Context& ctx, Extent2D<F32> available) {
    auto& s = *impl;
    s.advance(ctx.time);
    if (!s.config.child || s.progress <= 0.0f) {
        return {FillLength(available.x), 0.0f};
    }
    const F32 gap = Px(ctx, s.config.gap);
    const auto child = measureChild(*s.config.child, ctx, {available.x, std::numeric_limits<F32>::infinity()});
    return {FillLength(available.x), FitLength(std::round((child.y + gap) * Smoothstep(s.progress)), available.y)};
}

void Reveal::layout(const Context& ctx) {
    auto& s = *impl;
    s.advance(ctx.time);
    if (s.wake.has_value()) {
        requestFrame(*s.wake);
    }
    if (!s.config.child) {
        return;
    }
    s.present(*this, *s.config.child, s.showing());
    if (!s.showing()) {
        return;
    }
    const Rect bounds = frame();
    const F32 gap = Px(ctx, s.config.gap);
    const auto child = measureChild(*s.config.child, ctx, {bounds.width, std::numeric_limits<F32>::infinity()});
    const F32 full = child.y + gap;
    const F32 offset = s.config.upward ? 0.0f : bounds.height - full;
    layoutChild(ctx, *s.config.child, {bounds.x, bounds.y + offset + gap, bounds.width, child.y}, bounds);
}

}  // namespace Jetstream::Sakura
