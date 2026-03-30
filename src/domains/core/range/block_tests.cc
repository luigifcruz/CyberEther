#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/core/range/block.hh"
#include "jetstream/domains/dsp/amplitude/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Range block creates with F32 upstream",
                 "[modules][range][block]") {
    Blocks::Window window;
    REQUIRE(flowgraph->blockCreate("range_win", window, {}) == Result::SUCCESS);

    TensorMap amplitudeInputs;
    amplitudeInputs["signal"].requested("range_win", "window");
    REQUIRE(flowgraph->blockCreate("range_amp", "amplitude", {}, amplitudeInputs) ==
            Result::SUCCESS);

    TensorMap rangeInputs;
    rangeInputs["signal"].requested("range_amp", "signal");

    Blocks::Range config;
    config.min = -120.0f;
    config.max = 0.0f;
    REQUIRE(flowgraph->blockCreate("range_block", config, rangeInputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("range_block")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("range_block")->outputs().count("signal") == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "Range block rejects invalid min/max",
                 "[modules][range][block][validation]") {
    Blocks::Window window;
    REQUIRE(flowgraph->blockCreate("range_bad_win", window, {}) == Result::SUCCESS);

    TensorMap amplitudeInputs;
    amplitudeInputs["signal"].requested("range_bad_win", "window");
    REQUIRE(flowgraph->blockCreate("range_bad_amp", "amplitude", {}, amplitudeInputs) ==
            Result::SUCCESS);

    TensorMap rangeInputs;
    rangeInputs["signal"].requested("range_bad_amp", "signal");

    Blocks::Range config;
    config.min = 1.0f;
    config.max = -1.0f;
    REQUIRE(flowgraph->blockCreate("range_bad", config, rangeInputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("range_bad")->state() == Block::State::Errored);
}
