#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "viewport/glfw/tools/wayland_scroll.hh"

using namespace Jetstream;

namespace {

using Source = Viewport::WaylandScroll::Source;

struct ScrollFeed {
    Viewport::WaylandScroll scroll;
    std::vector<Render::InputQueue::Wheel> events;
    F64 now = 1.0;
    bool fresh = true;

    void frame() { scroll.frame(now, fresh, [this](const auto& event) { events.push_back(event); }); }
    void step(F64 time) {
        now = time;
        scroll.step(now, [this](const auto& event) { events.push_back(event); });
    }
    void cancel() { scroll.cancel([this](const auto& event) { events.push_back(event); }); }

    void finger(U32 time, F32 distance) {
        scroll.source = Source::Finger;
        scroll.timestamp = time;
        scroll.axes[0].distance = distance;
        frame();
    }

    void lift(U32 time) {
        scroll.timestamp = time;
        scroll.axes[0].stopped = true;
        frame();
    }

    void fling() {
        finger(100, 10.0f);
        finger(110, 10.0f);
        finger(120, 10.0f);
        lift(125);
    }

    U64 count(ScrollPhase phase) const {
        U64 result = 0;
        for (const auto& event : events) {
            result += event.phase == phase ? 1 : 0;
        }
        return result;
    }
};

}  // namespace

TEST_CASE("Wayland finger frames preserve subpixel distance", "[core][render][wayland-scroll]") {
    ScrollFeed feed;
    feed.scroll.source = Source::Finger;
    feed.scroll.axes[0].distance = 0.25f;
    feed.scroll.axes[1].distance = -1.5f;
    feed.frame();
    REQUIRE(feed.events.size() == 1);
    const auto& event = feed.events.front();
    CHECK(event.delta.x == Catch::Approx(0.15f));
    CHECK(event.delta.y == Catch::Approx(-0.025f));
    CHECK(event.phase == ScrollPhase::Began);
    CHECK(event.precise);

    feed.frame();
    CHECK(feed.events.size() == 1);
    feed.scroll.axes[0].distance = 2.5f;
    feed.frame();
    REQUIRE(feed.events.size() == 2);
    CHECK(feed.events.back().phase == ScrollPhase::Changed);
    CHECK(feed.events.back().delta.x == 0.0f);
}

TEST_CASE("Wayland waits for every scrolling axis to stop before releasing", "[core][render][wayland-scroll]") {
    ScrollFeed feed;
    feed.scroll.source = Source::Finger;
    feed.scroll.axes[0].distance = 10.0f;
    feed.scroll.axes[1].distance = 10.0f;
    feed.frame();
    feed.scroll.axes[0].stopped = true;
    feed.scroll.timestamp = 20;
    feed.frame();
    CHECK(feed.events.size() == 1);
    feed.scroll.axes[1].distance = 1.5f;
    feed.scroll.axes[1].stopped = true;
    feed.scroll.timestamp = 30;
    feed.frame();
    REQUIRE(feed.events.size() >= 3);
    CHECK(feed.events[1].phase == ScrollPhase::Changed);
    CHECK(feed.events[1].delta.x == Catch::Approx(-0.15f));
    CHECK(feed.events[2].phase == ScrollPhase::Ended);
    CHECK(feed.count(ScrollPhase::Ended) == 1);
}

TEST_CASE("Wayland high-resolution wheels preserve fractional notches without duplicate distances",
          "[core][render][wayland-scroll]") {
    ScrollFeed feed;
    feed.scroll.source = Source::Wheel;
    feed.scroll.axes[0] = {.distance = 10.0f, .discrete = 1, .value120 = 15};
    feed.scroll.axes[1] = {.distance = -10.0f, .discrete = -1, .value120 = -30};
    feed.frame();
    REQUIRE(feed.events.size() == 1);
    CHECK(feed.events[0].delta.y == Catch::Approx(-0.125f));
    CHECK(feed.events[0].delta.x == Catch::Approx(0.25f));
    CHECK(feed.events[0].phase == ScrollPhase::None);
    // The precise flag denotes point units, not fractional notch units.
    CHECK_FALSE(feed.events[0].precise);

    feed.scroll.source = Source::Wheel;
    feed.scroll.axes[0] = {.distance = 8.0f, .discrete = 1};
    feed.frame();
    CHECK(feed.events.back().delta.y == -1.0f);
    feed.scroll.axes[0].distance = 1.25f;
    feed.frame();
    CHECK(feed.events.back().delta.y == Catch::Approx(-0.125f));
}

TEST_CASE("Wayland continuous sources stay precise without fabricating finger release or inertia",
          "[core][render][wayland-scroll]") {
    ScrollFeed feed;
    feed.scroll.source = Source::Continuous;
    feed.scroll.axes[0].distance = 0.5f;
    feed.scroll.axes[0].stopped = true;
    feed.frame();
    REQUIRE(feed.events.size() == 1);
    CHECK(feed.events[0].precise);
    CHECK(feed.events[0].delta.y == Catch::Approx(-0.05f));
    CHECK(feed.events[0].phase == ScrollPhase::None);
    feed.step(1.1);
    CHECK(feed.events.size() == 1);
}

TEST_CASE("Wayland finger lifts coast with decaying momentum that ends once", "[core][render][wayland-scroll]") {
    ScrollFeed feed;
    feed.fling();
    REQUIRE(feed.events.size() == 5);
    CHECK(feed.events[3].phase == ScrollPhase::Ended);
    CHECK(feed.events[4].phase == ScrollPhase::Momentum);
    CHECK(feed.events[4].delta.y == 0.0f);
    F32 previous = -100.0f;
    for (U32 i = 1; i <= 1000 && feed.count(ScrollPhase::MomentumEnded) == 0; ++i) {
        const U64 before = feed.events.size();
        feed.step(1.0 + i * 0.003);
        for (U64 index = before; index < feed.events.size(); ++index) {
            const auto& event = feed.events[index];
            CHECK(event.precise);
            if (event.phase == ScrollPhase::Momentum) {
                CHECK(event.delta.y > previous);
                CHECK(event.delta.y < 0.0f);
                previous = event.delta.y;
            }
        }
    }
    CHECK(feed.count(ScrollPhase::MomentumEnded) == 1);
    const U64 settled = feed.events.size();
    feed.step(feed.now + 1.0);
    feed.cancel();
    CHECK(feed.events.size() == settled);
}

TEST_CASE("Wayland finger movement after a pause releases without momentum", "[core][render][wayland-scroll]") {
    ScrollFeed feed;
    feed.finger(100, 10.0f);
    feed.finger(110, 10.0f);
    feed.lift(3100);
    CHECK(feed.events.back().phase == ScrollPhase::Ended);
    feed.step(1.1);
    CHECK(feed.count(ScrollPhase::Momentum) == 0);
    CHECK(feed.count(ScrollPhase::MomentumEnded) == 0);
}

TEST_CASE("Wayland cancellations end each gesture or fling exactly once", "[core][render][wayland-scroll]") {
    ScrollFeed feed;
    SECTION("during the gesture") {
        feed.finger(100, 10.0f);
        feed.finger(110, 10.0f);
        feed.cancel();
        CHECK(feed.events.back().phase == ScrollPhase::Ended);
        CHECK(feed.count(ScrollPhase::Ended) == 1);
        feed.lift(120);
        CHECK(feed.count(ScrollPhase::Ended) == 1);
    }
    SECTION("while coasting") {
        feed.fling();
        feed.step(1.01);
        feed.cancel();
        CHECK(feed.events.back().phase == ScrollPhase::MomentumEnded);
        CHECK(feed.count(ScrollPhase::MomentumEnded) == 1);
    }
    const U64 settled = feed.events.size();
    feed.cancel();
    feed.step(feed.now + 0.01);
    CHECK(feed.events.size() == settled);
}

TEST_CASE("New Wayland axis activity stops a fling before more travel", "[core][render][wayland-scroll]") {
    ScrollFeed feed;
    feed.fling();
    SECTION("a new finger gesture") {
        feed.finger(200, 10.0f);
        REQUIRE(feed.events.size() == 7);
        CHECK(feed.events[5].phase == ScrollPhase::MomentumEnded);
        CHECK(feed.events[6].phase == ScrollPhase::Began);
    }
    SECTION("a wheel notch") {
        feed.scroll.source = Source::Wheel;
        feed.scroll.axes[0].value120 = 120;
        feed.frame();
        REQUIRE(feed.events.size() == 7);
        CHECK(feed.events[5].phase == ScrollPhase::MomentumEnded);
        CHECK(feed.events[6].phase == ScrollPhase::None);
    }
    SECTION("pointer motion alone keeps coasting") {
        feed.frame();
        CHECK(feed.events.size() == 5);
        feed.step(1.01);
        CHECK(feed.events.back().phase == ScrollPhase::Momentum);
        return;
    }
    feed.step(1.01);
    CHECK(feed.count(ScrollPhase::Momentum) == 1);
}

TEST_CASE("Wayland source switches cannot launch a fling", "[core][render][wayland-scroll]") {
    ScrollFeed feed;
    feed.finger(100, 10.0f);
    feed.finger(110, 10.0f);
    feed.scroll.source = Source::Wheel;
    feed.scroll.axes[0].value120 = 30;
    feed.frame();
    REQUIRE(feed.events.size() == 4);
    CHECK(feed.events[2].phase == ScrollPhase::Ended);
    CHECK(feed.events[3].delta.y == Catch::Approx(-0.25f));
    CHECK(feed.events[3].phase == ScrollPhase::None);
    feed.lift(120);
    feed.step(1.01);
    CHECK(feed.count(ScrollPhase::Momentum) == 0);
}

TEST_CASE("A stalled Wayland dispatch never starts a stale fling", "[core][render][wayland-scroll]") {
    ScrollFeed feed;
    SECTION("stall before the lift") {
        feed.finger(100, 10.0f);
        feed.finger(110, 10.0f);
        feed.fresh = false;
        feed.lift(115);
    }
    SECTION("stall mid-gesture") {
        feed.finger(100, 10.0f);
        feed.fresh = false;
        feed.finger(110, 10.0f);
        CHECK(feed.events.back().phase == ScrollPhase::Changed);
        feed.fresh = true;
        feed.finger(120, 10.0f);
        feed.lift(125);
    }
    CHECK(feed.events.back().phase == ScrollPhase::Ended);
    CHECK(feed.count(ScrollPhase::Momentum) == 0);

    feed.fresh = true;
    feed.events.clear();
    feed.fling();
    CHECK(feed.events.back().phase == ScrollPhase::Momentum);
}

TEST_CASE("A long gap between fling steps ends the fling without replaying it", "[core][render][wayland-scroll]") {
    ScrollFeed feed;
    feed.fling();
    feed.step(2.0);
    REQUIRE(feed.events.size() == 6);
    CHECK(feed.events.back().phase == ScrollPhase::MomentumEnded);
}
