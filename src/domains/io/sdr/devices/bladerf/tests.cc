#include <catch2/catch_test_macros.hpp>

#include <array>
#include <vector>

#include "domains/io/sdr/devices/bladerf/tools/quick_tune_budget.hh"
#include "domains/io/sdr/tools/samples.hh"

using namespace Jetstream;

TEST_CASE("bladeRF quick-tune budgets reuse plans and fall back before exhaustion",
          "[devices][sdr][bladerf][budget]") {
    using Decision = Radio::BladeRfQuickTuneBudget::Decision;
    Radio::BladeRfQuickTuneBudget budget(256);
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

    Radio::BladeRfQuickTuneBudget unbounded;
    REQUIRE(unbounded.plan(std::vector<F64>(1000, 1.0)) == Decision::Build);
    REQUIRE_FALSE(unbounded.remaining().has_value());
}

TEST_CASE("bladeRF dual IQ conversion preserves channel order and signed values",
          "[devices][sdr][bladerf][multi-channel]") {
    const std::array<I16, 12> raw{2047, -2048, 1024, -1024, 0, 204, -100, 512,
                                 -512, 256, 2048, 0};
    std::array<CF32, 3> first;
    std::array<CF32, 3> second;
    Radio::ConvertI16Channels(raw.data(), first.data(), second.data(), 3, 2048.0f);
    REQUIRE(first[0] == CF32{2047.0f / 2048.0f, -1.0f});
    REQUIRE(second[0] == CF32{0.5f, -0.5f});
    REQUIRE(first[1] == CF32{0.0f, 204.0f / 2048.0f});
    REQUIRE(second[1] == CF32{-100.0f / 2048.0f, 0.25f});
    REQUIRE(first[2] == CF32{-0.25f, 0.125f});
    REQUIRE(second[2] == CF32{1.0f, 0.0f});
}
