#include <catch2/catch_test_macros.hpp>

#include <string>
#include "jetstream/domains/dsp/filter_engine/block.hh"
#include "jetstream/domains/dsp/filter_taps/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Filter engine chain", "[modules][dsp][filter_engine]") {
    REQUIRE(flowgraph->blockCreate("taps_signal", "filter_taps", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("taps_filter", "filter_taps", {}, {}) == Result::SUCCESS);

    TensorMap engineInputs;
    engineInputs["signal"] = {"taps_signal", "coeffs", {}};
    engineInputs["filter"] = {"taps_filter", "coeffs", {}};
    REQUIRE(flowgraph->blockCreate("engine1", "filter_engine", {}, engineInputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("engine1")->state() == Block::State::Created);

    SECTION("disconnecting filter input marks engine incomplete") {
        auto result = flowgraph->blockDisconnect("engine1", "filter");
        REQUIRE((result == Result::SUCCESS || result == Result::INCOMPLETE));
        REQUIRE(flowgraph->blockList().at("engine1")->state() == Block::State::Incomplete);
    }

    SECTION("reconnecting filter input restores engine") {
        flowgraph->blockDisconnect("engine1", "filter");
        REQUIRE(flowgraph->blockList().at("engine1")->state() == Block::State::Incomplete);

        auto result = flowgraph->blockConnect("engine1", "filter", "taps_filter", "coeffs");
        REQUIRE(result == Result::SUCCESS);
        REQUIRE(flowgraph->blockList().at("engine1")->state() == Block::State::Created);
    }
}
