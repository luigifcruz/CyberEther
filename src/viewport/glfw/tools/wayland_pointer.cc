#include "wayland_pointer.hh"
#include "wayland_scroll.hh"

#include "jetstream/platform.hh"

#include <algorithm>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>

#include <wayland-client-core.h>

namespace {

struct WaylandClient {
    decltype(&wl_proxy_marshal_flags) proxyMarshalFlags = nullptr;
    decltype(&wl_proxy_add_listener) proxyAddListener = nullptr;
    decltype(&wl_proxy_destroy) proxyDestroy = nullptr;
    decltype(&wl_proxy_get_version) proxyGetVersion = nullptr;
    decltype(&wl_display_roundtrip) displayRoundtrip = nullptr;
};

WaylandClient waylandClient;

bool LoadWaylandClient() {
    static const bool loaded = [] {
        std::string error;
        void* handle = Jetstream::Platform::OpenDynamicLibrary(
            "libwayland-client.so.0", Jetstream::Platform::DynamicLibraryVisibility::Local, error);
        if (!handle) {
            return false;
        }
        const auto resolve = [&](auto& target, const char* symbol) {
            target = reinterpret_cast<std::remove_reference_t<decltype(target)>>(
                Jetstream::Platform::LoadDynamicLibrarySymbol(handle, symbol, error));
            return target != nullptr;
        };
        if (resolve(waylandClient.proxyMarshalFlags, "wl_proxy_marshal_flags") &&
            resolve(waylandClient.proxyAddListener, "wl_proxy_add_listener") &&
            resolve(waylandClient.proxyDestroy, "wl_proxy_destroy") &&
            resolve(waylandClient.proxyGetVersion, "wl_proxy_get_version") &&
            resolve(waylandClient.displayRoundtrip, "wl_display_roundtrip")) {
            return true;
        }
        waylandClient = {};
        Jetstream::Platform::CloseDynamicLibrary(handle);
        return false;
    }();
    return loaded;
}

}  // namespace

#define wl_proxy_marshal_flags waylandClient.proxyMarshalFlags
#define wl_proxy_add_listener waylandClient.proxyAddListener
#define wl_proxy_destroy waylandClient.proxyDestroy
#define wl_proxy_get_version waylandClient.proxyGetVersion
#ifndef wl_display_roundtrip
#define wl_display_roundtrip waylandClient.displayRoundtrip
#endif

#include "wayland-protocol-client.h"

#include "pointer-gestures-client.h"

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_WAYLAND
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

namespace Jetstream::Viewport {

struct WaylandPointer::State {
    State(GLFWwindow* window, std::function<void(const Scroll&)> onScroll, std::function<void(const Touch&)> onTouch)
        : onScroll(std::move(onScroll)), onTouch(std::move(onTouch)) {
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
        onTouch = {};
        releasePointer();
        releaseTouch();
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
    std::function<void(const Touch&)> onTouch;
    wl_display* display = nullptr;
    wl_surface* surface = nullptr;
    wl_registry* registry = nullptr;
    wl_seat* seat = nullptr;
    wl_pointer* pointer = nullptr;
    wl_touch* touch = nullptr;
    zwp_pointer_gestures_v1* gestures = nullptr;
    zwp_pointer_gesture_hold_v1* hold = nullptr;
    U32 seatName = 0;
    U32 gesturesName = 0;
    bool inside = false;
    Extent2D<F32> position = {0.0f, 0.0f};
    Extent2D<F32> scrollPosition = {0.0f, 0.0f};
    WaylandScroll scroll;
    F64 polledAt = glfwGetTime();

    void emit(const Render::InputQueue::Wheel& event) {
        if (event.phase == ScrollPhase::Began || event.phase == ScrollPhase::None) {
            scrollPosition = position;
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

    void emitTouch(Touch event) {
        event.momentumAllowed = glfwGetTime() - polledAt <= 0.25;
        if (onTouch) {
            onTouch(event);
        }
    }

    void cancelTouch() {
        emitTouch({.type = Touch::Type::Cancel});
    }

    void releaseTouch() {
        cancelTouch();
        if (touch) {
            wl_touch_release(touch);
            touch = nullptr;
        }
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
            self.releaseTouch();
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
        if (capabilities & WL_SEAT_CAPABILITY_TOUCH) {
            if (!self.touch && self.onTouch) {
                self.touch = wl_seat_get_touch(seat);
                wl_touch_add_listener(self.touch, &TouchListener, data);
            }
        } else {
            self.releaseTouch();
        }
    }

    static void TouchDown(void* data, wl_touch*, U32, U32 time, wl_surface* surface, I32 id,
                          wl_fixed_t x, wl_fixed_t y) {
        auto& self = Self(data);
        if (surface != self.surface) {
            return;
        }
        self.cancel();
        self.emitTouch({
            .type = Touch::Type::Down,
            .id = id,
            .position = {static_cast<F32>(wl_fixed_to_double(x)), static_cast<F32>(wl_fixed_to_double(y))},
            .timestamp = time,
        });
    }

    static void TouchUp(void* data, wl_touch*, U32, U32 time, I32 id) {
        Self(data).emitTouch({.type = Touch::Type::Up, .id = id, .timestamp = time});
    }

    static void TouchMotion(void* data, wl_touch*, U32 time, I32 id, wl_fixed_t x, wl_fixed_t y) {
        Self(data).emitTouch({
            .type = Touch::Type::Motion,
            .id = id,
            .position = {static_cast<F32>(wl_fixed_to_double(x)), static_cast<F32>(wl_fixed_to_double(y))},
            .timestamp = time,
        });
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
            const F64 now = glfwGetTime();
            self.scroll.frame(now, now - self.polledAt <= 0.25, [&self](const auto& event) { self.emit(event); });
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
    static constexpr wl_touch_listener TouchListener = {
        .down = TouchDown, .up = TouchUp, .motion = TouchMotion,
        .frame = [](void* data, wl_touch*) { Self(data).emitTouch({.type = Touch::Type::Frame}); },
        .cancel = [](void* data, wl_touch*) { Self(data).cancelTouch(); },
        .shape = [](void*, wl_touch*, I32, wl_fixed_t, wl_fixed_t) {},
        .orientation = [](void*, wl_touch*, I32, wl_fixed_t) {},
    };
    static constexpr zwp_pointer_gesture_hold_v1_listener HoldListener = {
        HoldBegin, [](void*, zwp_pointer_gesture_hold_v1*, U32, U32, I32) {},
    };
};

WaylandPointer::WaylandPointer(GLFWwindow* window, std::function<void(const Scroll&)> onScroll,
                               std::function<void(const Touch&)> onTouch) {
    if (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND && LoadWaylandClient()) {
        state = std::make_unique<State>(window, std::move(onScroll), std::move(onTouch));
    }
}

WaylandPointer::~WaylandPointer() = default;

bool WaylandPointer::available() const {
    return state && state->pointer;
}

void WaylandPointer::focus(bool focused) {
    if (state && !focused) {
        state->cancel();
        state->cancelTouch();
    }
}

void WaylandPointer::poll() {
    if (state) {
        state->polledAt = glfwGetTime();
        state->scroll.step(state->polledAt, [this](const auto& event) { state->emit(event); });
    }
}

}  // namespace Jetstream::Viewport
