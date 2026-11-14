#include "wayland_pointer.hh"
#include "wayland_scroll.hh"

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_WAYLAND
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <wayland-client.h>

#include "pointer-gestures-client.h"

#include <algorithm>
#include <cstring>
#include <utility>

namespace Jetstream::Viewport {

struct WaylandPointer::State {
    explicit State(GLFWwindow* window, std::function<void(const Scroll&)> callback) : onScroll(std::move(callback)) {
        display = glfwGetWaylandDisplay();
        surface = glfwGetWaylandWindow(window);
        registry = wl_display_get_registry(display);
        wl_registry_add_listener(registry, &RegistryListener, this);
        // Shares GLFW's display and event queue.
        wl_display_roundtrip(display);
        wl_display_roundtrip(display);
    }

    ~State() {
        onScroll = {};
        releasePointer();
        if (gestures) {
            zwp_pointer_gestures_v1_release(gestures);
        }
        if (seat) {
            wl_seat_release(seat);
        }
        if (registry) {
            wl_registry_destroy(registry);
        }
    }

    std::function<void(const Scroll&)> onScroll;
    wl_display* display = nullptr;
    wl_surface* surface = nullptr;
    wl_registry* registry = nullptr;
    wl_seat* seat = nullptr;
    wl_pointer* pointer = nullptr;
    zwp_pointer_gestures_v1* gestures = nullptr;
    zwp_pointer_gesture_hold_v1* hold = nullptr;
    U32 seatName = 0;
    U32 gesturesName = 0;
    bool inside = false;
    Extent2D<F32> position = {0.0f, 0.0f};
    Extent2D<F32> scrollPosition = {0.0f, 0.0f};
    WaylandScroll scroll;
    F64 polledAt = glfwGetTime();
    bool momentumAllowed = true;

    void emit(Render::InputQueue::Wheel event) {
        if (event.phase == ScrollPhase::Began || event.phase == ScrollPhase::None) {
            scrollPosition = position;
        }
        if (event.synthesizeMomentum) {
            if (event.phase == ScrollPhase::Began) {
                momentumAllowed = true;
            }
            momentumAllowed = momentumAllowed && glfwGetTime() - polledAt <= 0.25;
            if (event.phase == ScrollPhase::Ended) {
                event.synthesizeMomentum = momentumAllowed;
            }
        }
        if (onScroll) {
            onScroll({event, scrollPosition});
        }
    }

    void cancel() {
        scroll.cancel([this](const auto& event) { emit(event); });
        scroll.discardFrame();
    }

    void releasePointer() {
        cancel();
        if (hold) {
            zwp_pointer_gesture_hold_v1_destroy(hold);
            hold = nullptr;
        }
        if (pointer) {
            wl_pointer_release(pointer);
            pointer = nullptr;
        }
        inside = false;
    }

    void createHold() {
        if (gestures && pointer && !hold) {
            hold = zwp_pointer_gestures_v1_get_hold_gesture(gestures, pointer);
            zwp_pointer_gesture_hold_v1_add_listener(hold, &HoldListener, this);
        }
    }

    static State& Self(void* data) { return *static_cast<State*>(data); }

    static void Global(void* data, wl_registry* registry, U32 name, const char* interface, U32 version) {
        auto& self = Self(data);
        if (std::strcmp(interface, wl_seat_interface.name) == 0 && !self.seatName) {
            self.seatName = name;
            if (version < 5) {
                return;
            }
            self.seat = static_cast<wl_seat*>(wl_registry_bind(registry, name, &wl_seat_interface,
                std::min<U32>(version, 8)));
            wl_seat_add_listener(self.seat, &SeatListener, data);
        } else if (std::strcmp(interface, zwp_pointer_gestures_v1_interface.name) == 0 &&
                   !self.gestures && version >= 3) {
            self.gesturesName = name;
            self.gestures = static_cast<zwp_pointer_gestures_v1*>(
                wl_registry_bind(registry, name, &zwp_pointer_gestures_v1_interface, 3));
            self.createHold();
        }
    }

    static void GlobalRemove(void* data, wl_registry*, U32 name) {
        auto& self = Self(data);
        if (self.seat && name == self.seatName) {
            self.releasePointer();
            wl_seat_release(self.seat);
            self.seat = nullptr;
        }
        if (self.gestures && name == self.gesturesName) {
            if (self.hold) {
                zwp_pointer_gesture_hold_v1_destroy(self.hold);
                self.hold = nullptr;
            }
            zwp_pointer_gestures_v1_release(self.gestures);
            self.gestures = nullptr;
        }
    }

    static void Capabilities(void* data, wl_seat* seat, U32 capabilities) {
        auto& self = Self(data);
        if (capabilities & WL_SEAT_CAPABILITY_POINTER) {
            if (!self.pointer) {
                self.pointer = wl_seat_get_pointer(seat);
                wl_pointer_add_listener(self.pointer, &PointerListener, data);
                self.createHold();
            }
        } else {
            self.releasePointer();
        }
    }

    static void Enter(void* data, wl_pointer*, U32, wl_surface* surface, wl_fixed_t x, wl_fixed_t y) {
        auto& self = Self(data);
        self.cancel();
        self.inside = surface == self.surface;
        self.position = {static_cast<F32>(wl_fixed_to_double(x)), static_cast<F32>(wl_fixed_to_double(y))};
    }

    static void Leave(void* data, wl_pointer*, U32, wl_surface*) {
        auto& self = Self(data);
        self.cancel();
        self.inside = false;
    }

    static void Motion(void* data, wl_pointer*, U32, wl_fixed_t x, wl_fixed_t y) {
        Self(data).position = {static_cast<F32>(wl_fixed_to_double(x)), static_cast<F32>(wl_fixed_to_double(y))};
    }

    static void Button(void* data, wl_pointer*, U32, U32, U32, U32 state) {
        if (state == WL_POINTER_BUTTON_STATE_PRESSED) {
            Self(data).cancel();
        }
    }

    static void Axis(void* data, wl_pointer*, U32 time, U32 axis, wl_fixed_t value) {
        auto& scroll = Self(data).scroll;
        if (axis < scroll.axes.size()) {
            scroll.axes[axis].distance += static_cast<F32>(wl_fixed_to_double(value));
            scroll.timestamp = time;
        }
    }

    static void Frame(void* data, wl_pointer*) {
        auto& self = Self(data);
        if (self.inside) {
            self.scroll.frame([&self](const auto& event) { self.emit(event); });
        } else {
            self.scroll.discardFrame();
        }
    }

    static void Source(void* data, wl_pointer*, U32 source) {
        auto& scroll = Self(data).scroll;
        switch (source) {
            case WL_POINTER_AXIS_SOURCE_FINGER: scroll.source = WaylandScroll::Source::Finger; break;
            case WL_POINTER_AXIS_SOURCE_CONTINUOUS: scroll.source = WaylandScroll::Source::Continuous; break;
            case WL_POINTER_AXIS_SOURCE_WHEEL: scroll.source = WaylandScroll::Source::Wheel; break;
            case WL_POINTER_AXIS_SOURCE_WHEEL_TILT: scroll.source = WaylandScroll::Source::Tilt; break;
            default: scroll.source = WaylandScroll::Source::Unknown; break;
        }
    }

    static void Stop(void* data, wl_pointer*, U32 time, U32 axis) {
        auto& scroll = Self(data).scroll;
        if (axis < scroll.axes.size()) {
            scroll.axes[axis].stopped = true;
            scroll.timestamp = time;
        }
    }

    static void Discrete(void* data, wl_pointer*, U32 axis, I32 value) {
        auto& scroll = Self(data).scroll;
        if (axis < scroll.axes.size()) {
            scroll.axes[axis].discrete = scroll.axes[axis].discrete.value_or(0) + value;
        }
    }

    static void Value120(void* data, wl_pointer*, U32 axis, I32 value) {
        auto& scroll = Self(data).scroll;
        if (axis < scroll.axes.size()) {
            scroll.axes[axis].value120 = scroll.axes[axis].value120.value_or(0) + value;
        }
    }

    static void HoldBegin(void* data, zwp_pointer_gesture_hold_v1*, U32, U32, wl_surface* surface, U32) {
        auto& self = Self(data);
        if (surface == self.surface) {
            self.cancel();
        }
    }

    static constexpr wl_registry_listener RegistryListener = {Global, GlobalRemove};
    static constexpr wl_seat_listener SeatListener = {Capabilities, [](void*, wl_seat*, const char*) {}};
    static constexpr wl_pointer_listener PointerListener = {
        .enter = Enter, .leave = Leave, .motion = Motion, .button = Button,
        .axis = Axis, .frame = Frame, .axis_source = Source, .axis_stop = Stop,
        .axis_discrete = Discrete, .axis_value120 = Value120,
    };
    static constexpr zwp_pointer_gesture_hold_v1_listener HoldListener = {
        HoldBegin, [](void*, zwp_pointer_gesture_hold_v1*, U32, U32, I32) {},
    };
};

WaylandPointer::WaylandPointer(GLFWwindow* window, std::function<void(const Scroll&)> onScroll) {
    if (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND) {
        state = std::make_unique<State>(window, std::move(onScroll));
    }
}

WaylandPointer::~WaylandPointer() = default;

bool WaylandPointer::available() const {
    return state && state->pointer;
}

void WaylandPointer::focus(bool focused) {
    if (state && !focused) {
        state->cancel();
    }
}

void WaylandPointer::poll() {
    if (state) {
        state->polledAt = glfwGetTime();
    }
}

}  // namespace Jetstream::Viewport
