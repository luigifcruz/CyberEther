#ifndef JETSTREAM_VIEWPORT_TOOLS_SCROLL_MOMENTUM_HH
#define JETSTREAM_VIEWPORT_TOOLS_SCROLL_MOMENTUM_HH

#include <algorithm>
#include <array>
#include <cmath>

#include "jetstream/types.hh"

namespace Jetstream::Viewport {

class ScrollMomentum {
 public:
    bool active() const { return remaining > 0.0; }

    void reset() {
        count = 0;
        velocity = {0.0f, 0.0f};
        remaining = 0.0;
    }

    void sample(U32 time, Extent2D<F32> delta) {
        if (!std::isfinite(delta.x) || !std::isfinite(delta.y)) {
            reset();
            return;
        }
        if (count && time - samples[count - 1].time > Window) {
            count = 0;
        }
        if (count && time == samples[count - 1].time) {
            samples[count - 1].delta.x += delta.x;
            samples[count - 1].delta.y += delta.y;
            return;
        }
        if (count == samples.size()) {
            std::move(samples.begin() + 1, samples.end(), samples.begin());
            --count;
        }
        samples[count++] = {time, delta};
    }

    bool release(U32 time, F64 decay = Decay) {
        decayRate = decay;
        velocity = {speed(time, true), speed(time, false)};
        const F64 magnitude = std::hypot(velocity.x, velocity.y);
        remaining = magnitude > StopSpeed ? std::log(magnitude / StopSpeed) / decayRate : 0.0;
        count = 0;
        return active();
    }

    Extent2D<F32> advance(F64 dt) {
        if (!active() || !(dt > 0.0)) {
            return {0.0f, 0.0f};
        }
        if (dt > 0.25) {
            reset();
            return {0.0f, 0.0f};
        }
        dt = std::min(dt, remaining);
        const F32 decay = static_cast<F32>(std::exp(-decayRate * dt));
        const F32 distance = static_cast<F32>(-std::expm1(-decayRate * dt) / decayRate);
        const Extent2D<F32> delta = {velocity.x * distance, velocity.y * distance};
        velocity.x *= decay;
        velocity.y *= decay;
        remaining -= dt;
        return delta;
    }

 private:
    static constexpr U32 Window = 80;
    static constexpr F64 Decay = 5.5;
    static constexpr F64 StopSpeed = 0.5;
    static constexpr F32 MaxSpeed = 800.0f;

    struct Sample {
        U32 time = 0;
        Extent2D<F32> delta = {0.0f, 0.0f};
    };

    std::array<Sample, 32> samples{};
    U64 count = 0;
    Extent2D<F32> velocity = {0.0f, 0.0f};
    F64 remaining = 0.0;
    F64 decayRate = Decay;

    F32 speed(U32 time, bool horizontal) const {
        if (count < 2 || time - samples[count - 1].time >= Window) {
            return 0.0f;
        }
        F32 distance = 0.0f;
        U32 duration = 0;
        for (U64 i = count - 1; i > 0; --i) {
            const U32 age = time - samples[i - 1].time;
            if (age > Window) {
                break;
            }
            const F32 delta = horizontal ? samples[i].delta.x : samples[i].delta.y;
            if (distance * delta < 0.0f) {
                break;
            }
            distance += delta;
            duration = age;
        }
        return duration ? std::clamp(distance * 1000.0f / duration, -MaxSpeed, MaxSpeed) : 0.0f;
    }
};

}  // namespace Jetstream::Viewport

#endif
