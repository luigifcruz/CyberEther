#ifndef JETSTREAM_VIEWPORT_GLFW_WAYLAND_SCROLL_HH
#define JETSTREAM_VIEWPORT_GLFW_WAYLAND_SCROLL_HH

#include <array>
#include <optional>

#include "jetstream/render/base/input.hh"

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
    void frame(Emit emit) {
        using Wheel = Render::InputQueue::Wheel;
        const Source kind = source.value_or(gesture ? Source::Finger : Source::Unknown);
        const bool finger = kind == Source::Finger;
        const bool precise = finger || kind == Source::Continuous;
        if (!finger && gesture) {
            cancel(emit);
        }
        Extent2D<F32> delta = {distance(axes[1], precise), distance(axes[0], precise)};
        if (delta.x != 0.0f || delta.y != 0.0f) {
            Wheel event{.delta = delta, .precise = precise};
            if (finger) {
                event.phase = gesture ? ScrollPhase::Changed : ScrollPhase::Began;
                event.explicitRelease = true;
                event.synthesizeMomentum = true;
                event.timestamp = timestamp;
                gesture = true;
                mayCoast = false;
                active[0] = active[0] || delta.y != 0.0f;
                active[1] = active[1] || delta.x != 0.0f;
            }
            emit(event);
        }
        for (U64 i = 0; i < axes.size(); ++i) {
            if (axes[i].stopped) {
                active[i] = false;
            }
        }
        if (gesture && !active[0] && !active[1] && (axes[0].stopped || axes[1].stopped)) {
            emit(Wheel{.phase = ScrollPhase::Ended, .precise = true, .explicitRelease = true,
                       .synthesizeMomentum = true, .timestamp = timestamp});
            gesture = false;
            mayCoast = true;
        }
        axes = {};
        source.reset();
    }

    template <typename Emit>
    void cancel(Emit emit) {
        if (gesture || mayCoast) {
            emit(Render::InputQueue::Wheel{
                .phase = gesture ? ScrollPhase::Ended : ScrollPhase::MomentumEnded,
                .precise = true, .explicitRelease = true});
        }
        gesture = false;
        mayCoast = false;
        active = {};
    }

    void discardFrame() {
        axes = {};
        source.reset();
    }

 private:
    bool gesture = false;
    bool mayCoast = false;
    std::array<bool, 2> active{};

    static F32 distance(const Axis& axis, bool precise) {
        constexpr F32 AxisUnitsPerStep = 10.0f;
        constexpr F32 PointsPerStep = 48.0f;
        constexpr F32 PointsPerScrollUnit = 10.0f;

        if (precise) {
            return -axis.distance * (PointsPerStep / AxisUnitsPerStep) / PointsPerScrollUnit;
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
