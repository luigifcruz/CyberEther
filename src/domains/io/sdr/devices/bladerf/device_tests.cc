#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "domains/io/sdr/devices/bladerf/quick_tune_budget.hh"

using namespace Jetstream;

TEST_CASE("bladeRF quick-tune budgets reuse plans and fall back before exhaustion",
          "[devices][sdr][bladerf]") {
    using Decision = Radio::QuickTuneBudget::Decision;
    Radio::QuickTuneBudget budget(256);
    const std::vector<F64> first(200, 1.0e9);
    std::vector<F64> second(100, 2.0e9);

    REQUIRE(budget.plan(first) == Decision::Build);
    budget.built(first);
    REQUIRE(budget.used() == 200);
    REQUIRE(budget.plan(first) == Decision::Reuse);
    REQUIRE(budget.plan(second) == Decision::Fallback);
    REQUIRE(budget.remaining() == 56);

    second.resize(50);
    REQUIRE(budget.plan(second) == Decision::Build);
    budget.failed(10);
    REQUIRE(budget.used() == 210);
    REQUIRE(budget.plan(first) == Decision::Fallback);

    Radio::QuickTuneBudget unbounded;
    REQUIRE(unbounded.plan(std::vector<F64>(1000, 1.0)) == Decision::Build);
    REQUIRE_FALSE(unbounded.remaining().has_value());
}
