#include <catch2/catch_test_macros.hpp>

#include <string>

#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter block creates with default config",
                 "[modules][dsp][filter][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("512");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("filter", "filter", {}, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("filter")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("filter")->outputs().contains("buffer"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter block rejects heads smaller than centers",
                 "[modules][dsp][filter][block][validation]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("512");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    Parser::Map badConfig;
    badConfig["heads"] = std::string("1");
    badConfig["center"] = std::string("[0, 400000]");

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("filter_bad", "filter", badConfig, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("filter_bad")->state() == Block::State::Errored);
}
