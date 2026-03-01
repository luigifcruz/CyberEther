#include <catch2/catch_test_macros.hpp>

#include <string>

#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Signal generator block creates with default config",
                 "[modules][dsp][signal_generator][block]") {
    REQUIRE(flowgraph->blockCreate("gen", "signal_generator", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("gen")->state() == Block::State::Created);

    const Tensor& out = flowgraph->blockList().at("gen")->outputs().at("signal").tensor;
    REQUIRE(out.rank() == 1);
    REQUIRE(out.shape(0) == 8192);
    REQUIRE(out.dtype() == DataType::F32);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Signal generator block recreates on signal type change",
                 "[modules][dsp][signal_generator][block][reconfigure]") {
    REQUIRE(flowgraph->blockCreate("gen", "signal_generator", {}, {}) == Result::SUCCESS);

    Parser::Map update;
    update["signalType"] = std::string("chirp");

    REQUIRE(flowgraph->blockReconfigure("gen", update) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("gen")->state() == Block::State::Created);
}
