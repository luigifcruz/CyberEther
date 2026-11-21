#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "render/sakura/detail/wheel_motion.hh"

using namespace Jetstream;
using Jetstream::Sakura::WheelMotion;

TEST_CASE("Mouse wheel motion preserves fractional input and the full distance of a burst",
          "[core][sakura][wheel-motion]") {
    WheelMotion motion;
    F32 position = 100.0f;
    for (int i = 0; i < 24; ++i) {
        motion.add(position, 0.125f, 1000.0f);
        const F32 next = motion.advance(position, 0.004, 1000.0f);
        CHECK(next >= position);
        CHECK(next < 103.0f);
        position = next;
    }
    CHECK(motion.advance(position, 0.12, 1000.0f) == Catch::Approx(103.0f));
    CHECK_FALSE(motion.active());
}

TEST_CASE("Mouse wheel easing follows elapsed time instead of the number of frames",
          "[core][sakura][wheel-motion]") {
    WheelMotion slow;
    WheelMotion fast;
    slow.add(0.0f, 48.0f, 1000.0f);
    fast.add(0.0f, 48.0f, 1000.0f);
    F32 slowPosition = 0.0f;
    F32 fastPosition = 0.0f;
    for (int i = 0; i < 3; ++i) {
        slowPosition = slow.advance(slowPosition, 0.02, 1000.0f);
    }
    for (int i = 0; i < 12; ++i) {
        fastPosition = fast.advance(fastPosition, 0.005, 1000.0f);
    }
    CHECK(slowPosition == Catch::Approx(fastPosition));
    CHECK(slowPosition > 0.0f);
    CHECK(slowPosition < 48.0f);
    CHECK(slow.advance(slowPosition, 0.3, 1000.0f) == 48.0f);
    CHECK_FALSE(slow.active());
}

TEST_CASE("Reversing a mouse wheel cancels outstanding travel in the old direction",
          "[core][sakura][wheel-motion]") {
    WheelMotion motion;
    motion.add(100.0f, 100.0f, 1000.0f);
    const F32 turningPoint = motion.advance(100.0f, 0.02, 1000.0f);
    motion.add(turningPoint, -3.5f, 1000.0f);
    const F32 next = motion.advance(turningPoint, 0.02, 1000.0f);
    CHECK(next < turningPoint);
    CHECK(motion.advance(next, 0.12, 1000.0f) == Catch::Approx(turningPoint - 3.5f));
}

TEST_CASE("Wheel animation clamps at boundaries without accumulating hidden overscroll",
          "[core][sakura][wheel-motion]") {
    WheelMotion motion;
    motion.add(90.0f, 1000.0f, 100.0f);
    CHECK(motion.advance(90.0f, 0.12, 100.0f) == 100.0f);
    motion.add(100.0f, 1000.0f, 100.0f);
    CHECK_FALSE(motion.active());
    motion.add(100.0f, -0.25f, 100.0f);
    CHECK(motion.advance(100.0f, 0.12, 100.0f) == 99.75f);
    motion.add(99.75f, -1000.0f, 100.0f);
    CHECK(motion.advance(99.75f, 0.12, 100.0f) == 0.0f);
}

TEST_CASE("Resize and cancellation cannot restore stale wheel destinations",
          "[core][sakura][wheel-motion]") {
    WheelMotion motion;
    motion.add(100.0f, 48.0f, 1000.0f);
    CHECK(motion.advance(100.0f, 0.01, 80.0f) == 80.0f);
    CHECK_FALSE(motion.active());
    motion.add(80.0f, 100.0f, 1000.0f);
    motion.cancel();
    CHECK(motion.advance(20.0f, 0.12, 1000.0f) == 20.0f);
    motion.add(20.0f, -0.5f, 1000.0f);
    CHECK(motion.advance(20.0f, 0.12, 1000.0f) == 19.5f);
}
