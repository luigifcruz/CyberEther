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
    REQUIRE(viewBlock("range_block").state == Block::State::Created);
    REQUIRE(viewBlock("range_block").outputs.count("signal") == 1);

    Parser::Map equalBounds;
    equalBounds["min"] = F32{-46.666664f};
    equalBounds["max"] = F32{-46.666664f};
    REQUIRE(flowgraph->blockReconfigure("range_block", equalBounds) == Result::SUCCESS);
    REQUIRE(viewBlock("range_block").state == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture, "Range block orders reversed min/max",
                 "[modules][range][block][reversed-bounds]") {
    Blocks::Window window;
    REQUIRE(flowgraph->blockCreate("range_reverse_win", window, {}) == Result::SUCCESS);

    TensorMap amplitudeInputs;
    amplitudeInputs["signal"].requested("range_reverse_win", "window");
    REQUIRE(flowgraph->blockCreate("range_reverse_amp", "amplitude", {}, amplitudeInputs) ==
            Result::SUCCESS);

    TensorMap rangeInputs;
    rangeInputs["signal"].requested("range_reverse_amp", "signal");

    Blocks::Range config;
    config.min = 1.0f;
    config.max = -1.0f;
    REQUIRE(flowgraph->blockCreate("range_reverse", config, rangeInputs) == Result::SUCCESS);
    REQUIRE(viewBlock("range_reverse").state == Block::State::Created);
}
