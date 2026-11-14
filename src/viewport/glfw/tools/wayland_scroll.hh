#ifndef JETSTREAM_VIEWPORT_GLFW_WAYLAND_SCROLL_HH
#define JETSTREAM_VIEWPORT_GLFW_WAYLAND_SCROLL_HH

#include <array>
#include <optional>

#include "jetstream/render/base/input.hh"
#include "../../tools/scroll_momentum.hh"

namespace Jetstream::Viewport {

class WaylandScroll {
 public:
    enum class Source { Unknown, Wheel, Finger, Continuous, Tilt };

    struct Axis {
        F32 distance = 0.0f;
        std::optional<I32> discrete;
        std::optional<I32> value120;
        bool stopped = false;
    };

    std::array<Axis, 2> axes{};  // Wayland order: vertical, horizontal.
    std::optional<Source> source;
    U32 timestamp = 0;

    template <typename Emit>
    void frame(F64 now, bool fresh, Emit emit) {
        using Wheel = Render::InputQueue::Wheel;
        if (coasting && touched()) {
            cancel(emit);
        }
        const Source kind = source.value_or(gesture ? Source::Finger : Source::Unknown);
        const bool finger = kind == Source::Finger;
        const bool precise = finger || kind == Source::Continuous;
        if (!finger && gesture) {
            cancel(emit);
        }
        allowed = gesture && allowed && fresh;
        Extent2D<F32> delta = {distance(axes[1], precise), distance(axes[0], precise)};
        if (delta.x != 0.0f || delta.y != 0.0f) {
            Wheel event{.delta = delta, .precise = precise};
            if (finger) {
                if (!gesture) {
                    momentum.reset();
                    allowed = fresh;
                }
                event.phase = gesture ? ScrollPhase::Changed : ScrollPhase::Began;
                gesture = true;
                active[0] = active[0] || delta.y != 0.0f;
                active[1] = active[1] || delta.x != 0.0f;
                momentum.sample(timestamp, delta);
            }
            emit(event);
        }
        for (U64 i = 0; i < axes.size(); ++i) {
            if (axes[i].stopped) {
                active[i] = false;
            }
        }
        if (gesture && !active[0] && !active[1] && (axes[0].stopped || axes[1].stopped)) {
            gesture = false;
            emit(Wheel{.phase = ScrollPhase::Ended, .precise = true});
            coasting = allowed && momentum.release(timestamp);
            if (coasting) {
                steppedAt = now;
                emit(Wheel{.phase = ScrollPhase::Momentum, .precise = true});
            }
        }
        axes = {};
        source.reset();
    }

    template <typename Emit>
    void step(F64 now, Emit emit) {
        using Wheel = Render::InputQueue::Wheel;
        if (!coasting) {
            return;
        }
        const Extent2D<F32> delta = momentum.advance(now - steppedAt);
        steppedAt = now;
        if (delta.x != 0.0f || delta.y != 0.0f) {
            emit(Wheel{.delta = delta, .phase = ScrollPhase::Momentum, .precise = true});
        }
        if (!momentum.active()) {
            coasting = false;
            emit(Wheel{.phase = ScrollPhase::MomentumEnded, .precise = true});
        }
    }

    template <typename Emit>
    void cancel(Emit emit) {
        if (gesture || coasting) {
            emit(Render::InputQueue::Wheel{
                .phase = gesture ? ScrollPhase::Ended : ScrollPhase::MomentumEnded, .precise = true});
        }
        gesture = false;
        coasting = false;
        allowed = false;
        active = {};
        momentum.reset();
    }

    void discardFrame() {
        axes = {};
        source.reset();
    }

 private:
    bool gesture = false;
    bool coasting = false;
    bool allowed = false;
    F64 steppedAt = 0.0;
    std::array<bool, 2> active{};
    ScrollMomentum momentum;

    bool touched() const {
        for (const auto& axis : axes) {
            if (axis.distance != 0.0f || axis.discrete || axis.value120 || axis.stopped) {
                return true;
            }
        }
        return false;
    }

    static F32 distance(const Axis& axis, bool precise) {
        constexpr F32 AxisUnitsPerStep = 10.0f;
        constexpr F32 PointsPerStep = 48.0f;
        constexpr F32 PointsPerScrollUnit = 10.0f;

        if (precise) {
            return -axis.distance / AxisUnitsPerStep * PointsPerStep / PointsPerScrollUnit;
        }
        if (axis.value120.has_value()) {
            return -static_cast<F32>(*axis.value120) / 120.0f;
        }
        if (axis.discrete.has_value()) {
            return -static_cast<F32>(*axis.discrete);
        }
        return -axis.distance / AxisUnitsPerStep;
    }
};

}  // namespace Jetstream::Viewport

#endif
