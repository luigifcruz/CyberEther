#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "viewport/glfw/tools/wayland_scroll.hh"

using namespace Jetstream;

namespace {

struct ScrollFeed {
    Viewport::WaylandScroll scroll;
    std::vector<Render::InputQueue::Wheel> events;

    void frame() { scroll.frame([this](const auto& event) { events.push_back(event); }); }
    void cancel() { scroll.cancel([this](const auto& event) { events.push_back(event); }); }
};

}  // namespace

TEST_CASE("Wayland finger frames preserve subpixel distance and native timestamps", "[core][render][wayland-scroll]") {
    ScrollFeed feed;
    feed.scroll.source = Viewport::WaylandScroll::Source::Finger;
    feed.scroll.axes[0].distance = 0.25f;
    feed.scroll.axes[1].distance = -1.5f;
    feed.scroll.timestamp = 1234;
    feed.frame();
    REQUIRE(feed.events.size() == 1);
    const auto& event = feed.events.front();
    CHECK(event.delta.x == Catch::Approx(0.72f));
    CHECK(event.delta.y == Catch::Approx(-0.12f));
    CHECK(event.phase == ScrollPhase::Began);
    CHECK(event.precise);
    CHECK(event.explicitRelease);
    CHECK(event.synthesizeMomentum);
    CHECK(event.timestamp == 1234);

    feed.frame();
    CHECK(feed.events.size() == 1);
    feed.scroll.axes[0].distance = 2.5f;
    feed.scroll.timestamp = 1244;
    feed.frame();
    REQUIRE(feed.events.size() == 2);
    CHECK(feed.events.back().phase == ScrollPhase::Changed);
    CHECK(feed.events.back().delta.x == 0.0f);
}

TEST_CASE("Wayland waits for every scrolling axis to stop before releasing", "[core][render][wayland-scroll]") {
    ScrollFeed feed;
    feed.scroll.source = Viewport::WaylandScroll::Source::Finger;
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
    REQUIRE(feed.events.size() == 3);
    CHECK(feed.events[1].phase == ScrollPhase::Changed);
    CHECK(feed.events[1].delta.x == Catch::Approx(-0.72f));
    CHECK(feed.events[2].phase == ScrollPhase::Ended);
    CHECK(feed.events[2].synthesizeMomentum);
    CHECK(feed.events[2].timestamp == 30);
    feed.frame();
    CHECK(feed.events.size() == 3);
}

TEST_CASE("Wayland high-resolution wheels preserve fractional notches without duplicate distances",
          "[core][render][wayland-scroll]") {
    ScrollFeed feed;
    feed.scroll.source = Viewport::WaylandScroll::Source::Wheel;
    feed.scroll.axes[0] = {.distance = 10.0f, .discrete = 1, .value120 = 15};
    feed.scroll.axes[1] = {.distance = -10.0f, .discrete = -1, .value120 = -30};
    feed.frame();
    REQUIRE(feed.events.size() == 1);
    CHECK(feed.events[0].delta.y == Catch::Approx(-0.125f));
    CHECK(feed.events[0].delta.x == Catch::Approx(0.25f));
    CHECK(feed.events[0].phase == ScrollPhase::None);
    CHECK_FALSE(feed.events[0].synthesizeMomentum);
    // The precise flag denotes point units, not fractional notch units.
    CHECK_FALSE(feed.events[0].precise);

    feed.scroll.source = Viewport::WaylandScroll::Source::Wheel;
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
    feed.scroll.source = Viewport::WaylandScroll::Source::Continuous;
    feed.scroll.axes[0].distance = 0.5f;
    feed.scroll.axes[0].stopped = true;
    feed.frame();
    REQUIRE(feed.events.size() == 1);
    CHECK(feed.events[0].precise);
    CHECK(feed.events[0].delta.y == Catch::Approx(-0.24f));
    CHECK(feed.events[0].phase == ScrollPhase::None);
    CHECK_FALSE(feed.events[0].synthesizeMomentum);
}

TEST_CASE("Wayland source switches and cancellation cannot launch a fling", "[core][render][wayland-scroll]") {
    ScrollFeed feed;
    feed.scroll.source = Viewport::WaylandScroll::Source::Finger;
    feed.scroll.axes[0].distance = 10.0f;
    feed.frame();
    feed.scroll.source = Viewport::WaylandScroll::Source::Wheel;
    feed.scroll.axes[0].value120 = 30;
    feed.frame();
    REQUIRE(feed.events.size() == 3);
    CHECK(feed.events[1].phase == ScrollPhase::Ended);
    CHECK_FALSE(feed.events[1].synthesizeMomentum);
    CHECK(feed.events[2].delta.y == Catch::Approx(-0.25f));
    CHECK(feed.events[2].phase == ScrollPhase::None);

    feed.scroll.source = Viewport::WaylandScroll::Source::Finger;
    feed.scroll.axes[0].distance = 10.0f;
    feed.frame();
    CHECK(feed.events.back().phase == ScrollPhase::Began);
    feed.cancel();
    CHECK(feed.events.back().phase == ScrollPhase::Ended);
    CHECK_FALSE(feed.events.back().synthesizeMomentum);
    feed.scroll.axes[0].stopped = true;
    feed.frame();
    CHECK_FALSE(feed.events.back().synthesizeMomentum);
}

TEST_CASE("Wayland recontact cancels released momentum even without further axis movement",
          "[core][render][wayland-scroll]") {
    ScrollFeed feed;
    feed.scroll.source = Viewport::WaylandScroll::Source::Finger;
    feed.scroll.axes[0].distance = 10.0f;
    feed.frame();
    feed.scroll.axes[0].stopped = true;
    feed.frame();
    CHECK(feed.events.back().phase == ScrollPhase::Ended);
    feed.cancel();
    CHECK(feed.events.back().phase == ScrollPhase::MomentumEnded);
    CHECK_FALSE(feed.events.back().synthesizeMomentum);
    const auto count = feed.events.size();
    feed.cancel();
    CHECK(feed.events.size() == count);
}
