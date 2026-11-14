#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "viewport/tools/scroll_momentum.hh"

using namespace Jetstream;

TEST_CASE("Momentum needs recent movement to start", "[core][render][scroll-momentum]") {
    Viewport::ScrollMomentum momentum;
    momentum.sample(100, {0.0f, -1.0f});
    momentum.sample(110, {0.0f, -1.0f});
    CHECK_FALSE(momentum.release(3100));
    momentum.sample(100, {0.0f, -1.0f});
    CHECK_FALSE(momentum.release(105));
}

TEST_CASE("Momentum distance is independent of the stepping rate", "[core][render][scroll-momentum]") {
    const auto travel = [](F64 hz) {
        Viewport::ScrollMomentum momentum;
        momentum.sample(100, {0.0f, -1.0f});
        momentum.sample(110, {0.0f, -1.0f});
        momentum.sample(120, {0.0f, -1.0f});
        REQUIRE(momentum.release(125));
        F32 distance = 0.0f;
        for (int i = 0; i < hz * 3 && momentum.active(); ++i) {
            distance += momentum.advance(1.0 / hz).y;
        }
        CHECK_FALSE(momentum.active());
        return distance;
    };
    const F32 reference = travel(60.0);
    CHECK(reference < -5.0f);
    CHECK(travel(144.0) == Catch::Approx(reference).margin(0.001f));
    CHECK(travel(333.0) == Catch::Approx(reference).margin(0.001f));
}

TEST_CASE("Momentum stops instead of replaying a long gap", "[core][render][scroll-momentum]") {
    Viewport::ScrollMomentum momentum;
    momentum.sample(100, {0.0f, -1.0f});
    momentum.sample(110, {0.0f, -1.0f});
    REQUIRE(momentum.release(115));
    CHECK(momentum.advance(1.0).y == 0.0f);
    CHECK_FALSE(momentum.active());
}

TEST_CASE("Native scroll timestamps wrap and direction reversals discard old velocity",
          "[core][render][scroll-momentum]") {
    Viewport::ScrollMomentum momentum;
    momentum.sample(0xfffffff0u, {0.0f, -1.0f});
    momentum.sample(0xfffffffau, {0.0f, -1.0f});
    momentum.sample(4, {0.0f, -1.0f});
    REQUIRE(momentum.release(8));
    CHECK(momentum.advance(0.01).y < 0.0f);
    momentum.reset();
    momentum.sample(10, {0.0f, -10.0f});
    momentum.sample(20, {0.0f, -10.0f});
    momentum.sample(30, {0.0f, 0.25f});
    REQUIRE(momentum.release(35));
    const F32 distance = momentum.advance(0.01).y;
    CHECK(distance > 0.0f);
    CHECK(distance < 0.25f);
    momentum.advance(1.0);
    CHECK_FALSE(momentum.active());
}

TEST_CASE("A gentler decay coasts further from the same release", "[core][render][scroll-momentum]") {
    const auto travel = [](F64 decay) {
        Viewport::ScrollMomentum momentum;
        momentum.sample(100, {0.0f, -1.0f});
        momentum.sample(110, {0.0f, -1.0f});
        REQUIRE(momentum.release(115, decay));
        F32 distance = 0.0f;
        for (int i = 0; i < 1000 && momentum.active(); ++i) {
            distance += momentum.advance(1.0 / 120.0).y;
        }
        return distance;
    };
    const F32 standard = travel(5.5);
    const F32 gentle = travel(4.5);
    CHECK(gentle < standard);
    CHECK(gentle == Catch::Approx(standard * 5.5f / 4.5f).epsilon(0.01));
}
