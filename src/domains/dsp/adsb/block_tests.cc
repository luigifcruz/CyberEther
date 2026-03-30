#include <catch2/catch_test_macros.hpp>

#include <string>

#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "ADS-B block creates with CF32 signal input",
                 "[modules][dsp][adsb][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("8192");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("adsb", "adsb", {}, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("adsb")->state() == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "ADS-B block requires signal input",
                 "[modules][dsp][adsb][block][validation]") {
    auto result = flowgraph->blockCreate("adsb_incomplete", "adsb", {}, {});
    REQUIRE((result == Result::SUCCESS || result == Result::INCOMPLETE));
    REQUIRE(flowgraph->blockList().at("adsb_incomplete")->state() ==
            Block::State::Incomplete);
}
