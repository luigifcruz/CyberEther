#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <limits>

#include "render/components/axis_ticks.hh"

using namespace Jetstream;
using namespace Jetstream::Render::Components;

TEST_CASE("Axis ticks land on round values",
          "[core][render][axis]") {
    const auto isMultiple = [](const F64 value, const F64 step) {
        const F64 k = value / step;
        return std::abs(k - std::round(k)) < 1e-6;
    };

    SECTION("61.44 MS/s around 100 MHz") {
        const auto ticks = AxisTicks::Compute(69.28, 130.72, 6.0);
        REQUIRE(ticks.step == Catch::Approx(10.0));
        REQUIRE(ticks.decimals == 0);
        REQUIRE(ticks.minorDivisions == 5);
        REQUIRE(ticks.values.size() == 7);
        REQUIRE(ticks.values.front() == Catch::Approx(70.0));
        REQUIRE(ticks.values.back() == Catch::Approx(130.0));
    }

    SECTION("Two steps use quarter minor ticks") {
        const auto ticks = AxisTicks::Compute(97.5, 102.5, 1.5);
        REQUIRE(ticks.step == Catch::Approx(2.0));
        REQUIRE(ticks.minorDivisions == 4);
        REQUIRE(ticks.values.size() == 3);
        REQUIRE(ticks.values[1] == Catch::Approx(100.0));
    }

    SECTION("Sub-unit steps get decimals") {
        const auto ticks = AxisTicks::Compute(99.0, 101.0, 0.3);
        REQUIRE(ticks.step == Catch::Approx(0.5));
        REQUIRE(ticks.decimals == 1);
        REQUIRE(ticks.values.size() == 3);

        const auto fine = AxisTicks::Compute(100.0012, 100.0019, 0.0001);
        REQUIRE(fine.step == Catch::Approx(0.0001));
        REQUIRE(fine.decimals == 4);
        for (const F64 value : fine.values) {
            REQUIRE(isMultiple(value, fine.step));
        }
    }

    SECTION("Ticks never sit on the frame edges") {
        const auto ticks = AxisTicks::Compute(0.0, 10.0, 2.0);
        REQUIRE(ticks.values.size() == 4);
        REQUIRE(ticks.values.front() == Catch::Approx(2.0));
        REQUIRE(ticks.values.back() == Catch::Approx(8.0));
    }

    SECTION("Negative ranges and gigahertz values") {
        const auto normalized = AxisTicks::Compute(-1.0, 1.0, 0.4);
        REQUIRE(normalized.step == Catch::Approx(0.5));
        REQUIRE(normalized.values.size() == 3);
        REQUIRE(normalized.values[1] == Catch::Approx(0.0));

        const auto wifi = AxisTicks::Compute(2419.28, 2480.72, 7.0);
        REQUIRE(wifi.step == Catch::Approx(10.0));
        REQUIRE(wifi.values.front() == Catch::Approx(2420.0));
        REQUIRE(wifi.values.back() == Catch::Approx(2480.0));
    }

    SECTION("Invalid ranges produce no ticks") {
        REQUIRE(AxisTicks::Compute(1.0, 1.0, 0.1).values.empty());
        REQUIRE(AxisTicks::Compute(2.0, 1.0, 0.1).values.empty());
        REQUIRE(AxisTicks::Compute(0.0, std::numeric_limits<F64>::infinity(), 0.1).values.empty());
        const auto fallback = AxisTicks::Compute(0.0, 10.0, 0.0);
        REQUIRE(fallback.step == Catch::Approx(1.0));
    }
}

TEST_CASE("Axis minor ticks fall between the major ticks",
          "[core][render][axis]") {
    const auto ticks = AxisTicks::Compute(97.5, 102.5, 1.5);
    const auto minors = ticks.minors(97.5, 102.5);
    REQUIRE(minors.size() == 6);
    REQUIRE(minors.front() == Catch::Approx(98.5));
    REQUIRE(minors.back() == Catch::Approx(101.5));
    for (const F64 value : minors) {
        const F64 k = value / ticks.step;
        REQUIRE(std::abs(k - std::round(k)) > 1e-6);
    }

    REQUIRE(AxisTicks{}.minors(0.0, 1.0).empty());
}

TEST_CASE("Axis ticks stop when the step is below the value precision",
          "[core][render][axis][regression]") {
    const F64 left = 1.0e14 - 0.05;
    const F64 right = 1.0e14 + 0.05;
    const auto ticks = AxisTicks::Compute(left, right, 0.01);
    REQUIRE(ticks.values.size() <= AxisTicks::MaxValues);
    for (U64 i = 1; i < ticks.values.size(); i++) {
        REQUIRE(ticks.values[i] > ticks.values[i - 1]);
    }

    const auto minors = ticks.minors(left, right);
    REQUIRE(minors.size() <= AxisTicks::MaxValues * ticks.minorDivisions);
    for (U64 i = 1; i < minors.size(); i++) {
        REQUIRE(minors[i] > minors[i - 1]);
    }
}
