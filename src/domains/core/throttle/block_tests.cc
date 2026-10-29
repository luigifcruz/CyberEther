#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/core/throttle/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Throttle block creates with non-default interval",
                 "[modules][throttle][block]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("thr_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("thr_src", "window");

    Blocks::Throttle config;
    config.intervalMs = 5;
    REQUIRE(flowgraph->blockCreate("thr_block", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("thr_block").state == Block::State::Created);
    REQUIRE(viewBlock("thr_block").outputs.count("buffer") == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "Throttle block rejects zero interval",
                 "[modules][throttle][block][validation]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("thr_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("thr_bad_src", "window");

    Blocks::Throttle config;
    config.intervalMs = 0;
    REQUIRE(flowgraph->blockCreate("thr_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("thr_bad").state == Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture, "Throttle block keeps whole and fractional millisecond intervals",
                 "[modules][throttle][block]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("thr_ms_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("thr_ms_src", "window");

    REQUIRE(flowgraph->blockCreate("thr_saved", "throttle", {{"intervalMs", U64{30}}}, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("thr_saved").state == Block::State::Created);
    REQUIRE(Parser::Get<F32>(viewBlock("thr_saved").config, "intervalMs") == 30.0f);

    REQUIRE(flowgraph->blockReconfigure("thr_saved", {{"intervalMs", 0.25f}}) == Result::SUCCESS);
    REQUIRE(viewBlock("thr_saved").state == Block::State::Created);
    REQUIRE(Parser::Get<F32>(viewBlock("thr_saved").config, "intervalMs") == 0.25f);
}
