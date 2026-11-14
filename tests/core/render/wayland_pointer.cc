#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_WAYLAND
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <wayland-client.h>
#include <wayland-server.h>
#include <sys/socket.h>

#include <memory>
#include <vector>

#include "pointer-gestures-server.h"

namespace {

wl_display* server = nullptr;
wl_display* client = nullptr;
wl_surface* surface = nullptr;
int platform = GLFW_PLATFORM_WAYLAND;
double clockTime = 1.0;

// Pump a real client/server protocol connection deterministically on one thread.
// The helper's GLFW accessors are the only stand-ins; all Wayland messages are real.
int Roundtrip(wl_display* display) {
    bool done = false;
    const wl_callback_listener listener = {
        [](void* data, wl_callback* callback, uint32_t) {
            *static_cast<bool*>(data) = true;
            wl_callback_destroy(callback);
        },
    };
    auto* callback = wl_display_sync(display);
    wl_callback_add_listener(callback, &listener, &done);
    while (!done) {
        wl_display_flush(display);
        wl_event_loop_dispatch(wl_display_get_event_loop(server), 0);
        wl_display_flush_clients(server);
        if (wl_display_dispatch(display) < 0) return -1;
    }
    return 0;
}

int TestGlfwPlatform() { return platform; }
double TestGlfwTime() { return clockTime; }
wl_display* TestGlfwDisplay() { return client; }
wl_surface* TestGlfwSurface(GLFWwindow*) { return surface; }

}  // namespace

#define glfwGetPlatform ::TestGlfwPlatform
#define glfwGetTime ::TestGlfwTime
#define glfwGetWaylandDisplay ::TestGlfwDisplay
#define glfwGetWaylandWindow ::TestGlfwSurface
#define wl_display_roundtrip ::Roundtrip
#include "../../../src/viewport/glfw/tools/wayland_pointer.cc"
#undef wl_display_roundtrip
#undef glfwGetWaylandWindow
#undef glfwGetWaylandDisplay
#undef glfwGetPlatform
#undef glfwGetTime

using namespace Jetstream;

namespace {

struct Compositor {
    wl_global* seatGlobal = nullptr;
    wl_resource* seat = nullptr;
    wl_resource* pointer = nullptr;
    wl_resource* content = nullptr;
    wl_resource* hold = nullptr;
    wl_compositor* compositor = nullptr;
    wl_registry* registry = nullptr;
    std::unique_ptr<Viewport::WaylandPointer> helper;
    std::vector<Viewport::WaylandPointer::Scroll> events;
    std::vector<uint32_t> boundSeatVersions;

    static void Destroy(wl_client*, wl_resource* resource) { wl_resource_destroy(resource); }

    static Compositor& Self(wl_resource* resource) {
        return *static_cast<Compositor*>(wl_resource_get_user_data(resource));
    }

    static void CreateSurface(wl_client* client, wl_resource* resource, uint32_t id) {
        auto& self = Self(resource);
        self.content = wl_resource_create(client, &wl_surface_interface, 1, id);
        static const struct wl_surface_interface implementation = {.destroy = Destroy};
        wl_resource_set_implementation(self.content, &implementation, &self, nullptr);
    }

    static void GetPointer(wl_client* client, wl_resource* resource, uint32_t id) {
        auto& self = Self(resource);
        self.pointer = wl_resource_create(client, &wl_pointer_interface, wl_resource_get_version(resource), id);
        static const struct wl_pointer_interface implementation = {
            .set_cursor = [](wl_client*, wl_resource*, uint32_t, wl_resource*, int32_t, int32_t) {},
            .release = Destroy,
        };
        wl_resource_set_implementation(self.pointer, &implementation, &self, [](wl_resource* resource) {
            auto& owner = Self(resource);
            if (owner.pointer == resource) owner.pointer = nullptr;
        });
    }

    wl_global* addSeat(uint32_t version) {
        return wl_global_create(server, &wl_seat_interface, version, this,
            [](wl_client* client, void* data, uint32_t version, uint32_t id) {
                auto& self = *static_cast<Compositor*>(data);
                self.boundSeatVersions.push_back(version);
                self.seat = wl_resource_create(client, &wl_seat_interface, version, id);
                static const struct wl_seat_interface implementation = {.get_pointer = GetPointer, .release = Destroy};
                wl_resource_set_implementation(self.seat, &implementation, data, nullptr);
                wl_seat_send_capabilities(self.seat, WL_SEAT_CAPABILITY_POINTER);
            });
    }

    explicit Compositor(uint32_t version = 8, bool gestures = false, uint32_t secondSeatVersion = 0) {
        platform = GLFW_PLATFORM_WAYLAND;
        clockTime = 1.0;
        server = wl_display_create();
        REQUIRE(server);
        int sockets[2];
        REQUIRE(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets) == 0);
        REQUIRE(wl_client_create(server, sockets[0]));
        client = wl_display_connect_to_fd(sockets[1]);
        REQUIRE(client);
        wl_global_create(server, &wl_compositor_interface, 1, this,
            [](wl_client* client, void* data, uint32_t version, uint32_t id) {
                auto* resource = wl_resource_create(client, &wl_compositor_interface, version, id);
                static const struct wl_compositor_interface implementation = {.create_surface = CreateSurface};
                wl_resource_set_implementation(resource, &implementation, data, nullptr);
            });
        seatGlobal = addSeat(version);
        if (secondSeatVersion) {
            addSeat(secondSeatVersion);
        }
        if (gestures) {
            wl_global_create(server, &zwp_pointer_gestures_v1_interface, 3, this,
                [](wl_client* client, void* data, uint32_t version, uint32_t id) {
                    auto* resource = wl_resource_create(client, &zwp_pointer_gestures_v1_interface, version, id);
                    static const struct zwp_pointer_gestures_v1_interface implementation = {
                        .release = Destroy,
                        .get_hold_gesture = [](wl_client* client, wl_resource* resource, uint32_t id, wl_resource*) {
                            auto& self = Self(resource);
                            self.hold = wl_resource_create(client, &zwp_pointer_gesture_hold_v1_interface, 3, id);
                            static const struct zwp_pointer_gesture_hold_v1_interface implementation = {.destroy = Destroy};
                            wl_resource_set_implementation(self.hold, &implementation, &self, nullptr);
                        },
                    };
                    wl_resource_set_implementation(resource, &implementation, data, nullptr);
                });
        }
        registry = wl_display_get_registry(client);
        static const wl_registry_listener listener = {
            [](void* data, wl_registry* registry, uint32_t name, const char* interface, uint32_t) {
                if (std::strcmp(interface, "wl_compositor") == 0) {
                    static_cast<Compositor*>(data)->compositor = static_cast<wl_compositor*>(
                        wl_registry_bind(registry, name, &wl_compositor_interface, 1));
                }
            },
            [](void*, wl_registry*, uint32_t) {},
        };
        wl_registry_add_listener(registry, &listener, this);
        REQUIRE(Roundtrip(client) == 0);
        REQUIRE(compositor);
        surface = wl_compositor_create_surface(compositor);
        REQUIRE(Roundtrip(client) == 0);
        helper = std::make_unique<Viewport::WaylandPointer>(nullptr,
            [this](const auto& scroll) { events.push_back(scroll); });
        REQUIRE(Roundtrip(client) == 0);
    }

    ~Compositor() {
        helper.reset();
        wl_surface_destroy(surface);
        wl_compositor_destroy(compositor);
        wl_registry_destroy(registry);
        wl_display_flush(client);
        wl_event_loop_dispatch(wl_display_get_event_loop(server), 0);
        wl_display_disconnect(client);
        wl_display_destroy_clients(server);
        wl_display_destroy(server);
        client = nullptr;
        server = nullptr;
        surface = nullptr;
    }

    void enter() {
        REQUIRE(pointer);
        wl_pointer_send_enter(pointer, 1, content, wl_fixed_from_double(12.5), wl_fixed_from_double(34.25));
        frame();
    }

    void frame() {
        wl_pointer_send_frame(pointer);
        REQUIRE(Roundtrip(client) == 0);
        helper->poll();
    }

    void finger(uint32_t time, double distance) {
        wl_pointer_send_axis_source(pointer, WL_POINTER_AXIS_SOURCE_FINGER);
        wl_pointer_send_axis(pointer, time, WL_POINTER_AXIS_VERTICAL_SCROLL, wl_fixed_from_double(distance));
        frame();
    }
};

}  // namespace

TEST_CASE("Native Wayland pointer receives precise frames and explicit lift", "[wayland][native][scroll]") {
    Compositor rig;
    REQUIRE(rig.helper->available());
    rig.enter();
    rig.finger(100, 0.25);
    rig.finger(110, 1.5);
    wl_pointer_send_motion(rig.pointer, 112, wl_fixed_from_int(200), wl_fixed_from_int(250));
    wl_pointer_send_axis_stop(rig.pointer, 115, WL_POINTER_AXIS_VERTICAL_SCROLL);
    rig.frame();
    REQUIRE(rig.events.size() == 4);
    CHECK(rig.events[0].position == Extent2D<F32>{12.5f, 34.25f});
    CHECK(rig.events[0].event.delta.y == Catch::Approx(-0.025f));
    CHECK(rig.events[0].event.phase == ScrollPhase::Began);
    CHECK(rig.events[1].event.phase == ScrollPhase::Changed);
    CHECK(rig.events[2].event.phase == ScrollPhase::Ended);
    CHECK(rig.events[2].position == rig.events[0].position);
    CHECK(rig.events[3].event.phase == ScrollPhase::Momentum);
}

TEST_CASE("Native Wayland flings coast on the platform poll and end once", "[wayland][native][scroll]") {
    Compositor rig;
    rig.enter();
    rig.finger(100, 10.0);
    rig.finger(110, 10.0);
    rig.finger(120, 10.0);
    wl_pointer_send_axis_stop(rig.pointer, 125, WL_POINTER_AXIS_VERTICAL_SCROLL);
    rig.frame();
    REQUIRE(rig.events.back().event.phase == ScrollPhase::Momentum);
    F32 travel = 0.0f;
    U64 ended = 0;
    for (U32 i = 0; i < 1000 && ended == 0; ++i) {
        clockTime += 0.003;
        const U64 before = rig.events.size();
        rig.helper->poll();
        for (U64 index = before; index < rig.events.size(); ++index) {
            const auto& packet = rig.events[index];
            CHECK(packet.position == Extent2D<F32>{12.5f, 34.25f});
            travel += packet.event.delta.y;
            ended += packet.event.phase == ScrollPhase::MomentumEnded ? 1 : 0;
        }
    }
    CHECK(ended == 1);
    CHECK(travel < -5.0f);
    const U64 settled = rig.events.size();
    clockTime += 0.1;
    rig.helper->poll();
    rig.helper->focus(false);
    CHECK(rig.events.size() == settled);
}

TEST_CASE("Native Wayland wheel uses value120 rather than its simultaneous axis value", "[wayland][native][scroll]") {
    Compositor rig;
    rig.enter();
    wl_pointer_send_axis_source(rig.pointer, WL_POINTER_AXIS_SOURCE_WHEEL);
    wl_pointer_send_axis_value120(rig.pointer, WL_POINTER_AXIS_VERTICAL_SCROLL, 15);
    wl_pointer_send_axis(rig.pointer, 100, WL_POINTER_AXIS_VERTICAL_SCROLL, wl_fixed_from_double(10.0));
    rig.frame();
    REQUIRE(rig.events.size() == 1);
    CHECK(rig.events[0].event.delta.y == Catch::Approx(-0.125f));
    CHECK(rig.events[0].event.phase == ScrollPhase::None);
}

TEST_CASE("Native Wayland leave and capability loss cancel instead of releasing momentum", "[wayland][native][scroll]") {
    Compositor rig;
    rig.enter();
    rig.finger(100, 10.0);
    SECTION("leave") {
        wl_pointer_send_leave(rig.pointer, 2, rig.content);
        rig.frame();
        rig.finger(110, 10.0);
    }
    SECTION("focus") { rig.helper->focus(false); }
    SECTION("button") {
        wl_pointer_send_button(rig.pointer, 2, 110, 0x110, WL_POINTER_BUTTON_STATE_PRESSED);
        rig.frame();
    }
    SECTION("capability loss") {
        wl_seat_send_capabilities(rig.seat, 0);
        REQUIRE(Roundtrip(client) == 0);
        CHECK_FALSE(rig.helper->available());
        wl_seat_send_capabilities(rig.seat, WL_SEAT_CAPABILITY_POINTER);
        REQUIRE(Roundtrip(client) == 0);
        REQUIRE(Roundtrip(client) == 0);
        CHECK(rig.helper->available());
    }
    SECTION("seat removal") {
        wl_global_remove(rig.seatGlobal);
        REQUIRE(Roundtrip(client) == 0);
        CHECK_FALSE(rig.helper->available());
    }
    REQUIRE(rig.events.size() == 2);
    CHECK(rig.events.back().event.phase == ScrollPhase::Ended);
    clockTime += 0.01;
    rig.helper->poll();
    CHECK(rig.events.size() == 2);
}

TEST_CASE("Native Wayland version five retains precise scrolling without high-resolution wheel metadata",
          "[wayland][native][scroll]") {
    Compositor rig(5);
    REQUIRE(rig.helper->available());
    rig.enter();
    rig.finger(100, 0.25);
    REQUIRE(rig.events.size() == 1);
    CHECK(rig.events[0].event.delta.y == Catch::Approx(-0.025f));
}

TEST_CASE("Native Wayland hold gestures cancel a fling without another scroll delta", "[wayland][native][scroll]") {
    Compositor rig(8, true);
    rig.enter();
    REQUIRE(rig.hold);
    rig.finger(100, 10.0);
    rig.finger(110, 10.0);
    wl_pointer_send_axis_stop(rig.pointer, 115, WL_POINTER_AXIS_VERTICAL_SCROLL);
    rig.frame();
    REQUIRE(rig.events.back().event.phase == ScrollPhase::Momentum);
    zwp_pointer_gesture_hold_v1_send_begin(rig.hold, 3, 120, rig.content, 2);
    REQUIRE(Roundtrip(client) == 0);
    REQUIRE(rig.events.size() == 5);
    CHECK(rig.events.back().event.phase == ScrollPhase::MomentumEnded);
    clockTime += 0.01;
    rig.helper->poll();
    CHECK(rig.events.size() == 5);
}

TEST_CASE("Old Wayland seats leave GLFW's fallback available", "[wayland][native][scroll]") {
    Compositor rig(4);
    CHECK_FALSE(rig.helper->available());
    CHECK(rig.events.empty());
}

TEST_CASE("Native Wayland scrolling follows GLFW's first seat even among mixed versions", "[wayland][native][scroll]") {
    SECTION("old first seat keeps GLFW fallback") {
        Compositor rig(4, false, 8);
        CHECK_FALSE(rig.helper->available());
        CHECK(rig.boundSeatVersions.empty());
        rig.addSeat(8);
        REQUIRE(Roundtrip(client) == 0);
        CHECK_FALSE(rig.helper->available());
    }
    SECTION("a newer second seat cannot replace the first") {
        Compositor rig(5, false, 8);
        REQUIRE(rig.helper->available());
        CHECK(rig.boundSeatVersions == std::vector<uint32_t>{5});
        rig.enter();
        rig.finger(100, 0.25);
        REQUIRE(rig.events.size() == 1);
        wl_global_remove(rig.seatGlobal);
        REQUIRE(Roundtrip(client) == 0);
        rig.addSeat(8);
        REQUIRE(Roundtrip(client) == 0);
        CHECK_FALSE(rig.helper->available());
        CHECK(rig.boundSeatVersions == std::vector<uint32_t>{5});
    }
}

TEST_CASE("A first gesture queued during a native dispatch stall cannot start stale momentum", "[wayland][native][scroll]") {
    Compositor rig;
    rig.enter();
    for (uint32_t time : {100, 110, 120}) {
        wl_pointer_send_axis_source(rig.pointer, WL_POINTER_AXIS_SOURCE_FINGER);
        wl_pointer_send_axis(rig.pointer, time, WL_POINTER_AXIS_VERTICAL_SCROLL, wl_fixed_from_int(10));
        wl_pointer_send_frame(rig.pointer);
    }
    wl_pointer_send_axis_stop(rig.pointer, 125, WL_POINTER_AXIS_VERTICAL_SCROLL);
    wl_pointer_send_frame(rig.pointer);
    clockTime += 1.0;
    REQUIRE(Roundtrip(client) == 0);
    rig.helper->poll();
    REQUIRE(rig.events.size() == 4);
    CHECK(rig.events.back().event.phase == ScrollPhase::Ended);
    rig.events.clear();
    rig.finger(1100, 10.0);
    rig.finger(1110, 10.0);
    wl_pointer_send_axis_stop(rig.pointer, 1115, WL_POINTER_AXIS_VERTICAL_SCROLL);
    rig.frame();
    REQUIRE(rig.events.size() == 4);
    CHECK(rig.events.back().event.phase == ScrollPhase::Momentum);
}

TEST_CASE("A native dispatch stall never invents a finger release mid-gesture", "[wayland][native][scroll]") {
    Compositor rig;
    rig.enter();
    rig.finger(100, 10.0);
    clockTime += 1.0;
    rig.finger(110, 10.0);
    REQUIRE(rig.events.size() == 2);
    CHECK(rig.events.back().event.phase == ScrollPhase::Changed);
    wl_pointer_send_axis_stop(rig.pointer, 115, WL_POINTER_AXIS_VERTICAL_SCROLL);
    rig.frame();
    REQUIRE(rig.events.size() == 3);
    CHECK(rig.events.back().event.phase == ScrollPhase::Ended);
}

int main(int argc, char** argv) {
    return Catch::Session().run(argc, argv);
}
