#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_WHEEL_MOTION_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_WHEEL_MOTION_HH

#include <jetstream/types.hh>

#include <algorithm>
#include <cmath>

namespace Jetstream::Sakura {

class WheelMotion {
 public:
    static constexpr F64 Duration = 0.12;

    bool active() const {
        return moving;
    }

    void cancel() {
        moving = false;
    }

    void add(F32 current, F32 delta, F32 maximum) {
        if (!std::isfinite(delta) || delta == 0.0f) {
            return;
        }
        maximum = std::max(0.0f, maximum);
        current = std::clamp(current, 0.0f, maximum);
        destination = moving ? std::clamp(destination, 0.0f, maximum) : current;
        if ((destination - current) * delta < 0.0f) {
            destination = current;
        }
        origin = current;
        destination = std::clamp(destination + delta, 0.0f, maximum);
        elapsed = 0.0;
        moving = destination != current;
    }

    F32 advance(F32 current, F64 dt, F32 maximum) {
        if (!moving) {
            return current;
        }
        maximum = std::max(0.0f, maximum);
        origin = std::clamp(origin, 0.0f, maximum);
        destination = std::clamp(destination, 0.0f, maximum);
        elapsed = std::clamp(elapsed + std::max(0.0, dt), 0.0, Duration);
        if (elapsed >= Duration || origin == destination) {
            moving = false;
            return destination;
        }
        const F64 remaining = 1.0 - elapsed / Duration;
        const auto progress = static_cast<F32>(1.0 - remaining * remaining * remaining);
        return origin + (destination - origin) * progress;
    }

 private:
    F32 origin = 0.0f;
    F32 destination = 0.0f;
    F64 elapsed = 0.0;
    bool moving = false;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_WHEEL_MOTION_HH
