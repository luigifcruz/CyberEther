#include <catch2/catch_test_macros.hpp>

#include <string>

#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Overlap-add block creates with both required inputs",
                 "[modules][dsp][overlap_add][block]") {
    Parser::Map bufferConfig;
    bufferConfig["signalDataType"] = std::string("CF32");
    bufferConfig["bufferSize"] = std::string("128");

    Parser::Map overlapConfig;
    overlapConfig["size"] = std::string("16");

    REQUIRE(flowgraph->blockCreate("buffer_src", "signal_generator", bufferConfig, {}) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("overlap_src", "window", overlapConfig, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("buffer_src", "signal");
    inputs["overlap"].requested("overlap_src", "window");

    REQUIRE(flowgraph->blockCreate("overlap_add", "overlap_add", {}, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("overlap_add")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("overlap_add")->outputs().contains("buffer"));
}
