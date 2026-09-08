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
    sourceConfig["sampleRate"] = std::string("2000000");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("adsb", "adsb", {}, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("adsb").state == Block::State::Created);
    REQUIRE(viewBlock("adsb").outputs.empty());
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "ADS-B block requires signal input",
                 "[modules][dsp][adsb][block][validation]") {
    auto result = flowgraph->blockCreate("adsb_incomplete", "adsb", {}, {});
    REQUIRE((result == Result::SUCCESS || result == Result::INCOMPLETE));
    REQUIRE(viewBlock("adsb_incomplete").state ==
            Block::State::Incomplete);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "ADS-B block delegates signal validation to its module",
                 "[modules][dsp][adsb][block][validation]") {
    SECTION("dtype") {
        Parser::Map sourceConfig;
        sourceConfig["signalDataType"] = std::string("F32");
        sourceConfig["bufferSize"] = std::string("8192");

        REQUIRE(flowgraph->blockCreate("adsb_dtype_src", "signal_generator",
                                       sourceConfig, {}) == Result::SUCCESS);

        TensorMap inputs;
        inputs["signal"].requested("adsb_dtype_src", "signal");
        REQUIRE(flowgraph->blockCreate("adsb_dtype_bad", "adsb", {}, inputs) ==
                Result::SUCCESS);
        REQUIRE(viewBlock("adsb_dtype_bad").state == Block::State::Errored);
        REQUIRE(viewBlock("adsb_dtype_bad").outputs.empty());
    }

    SECTION("metadata") {
        Parser::Map sourceConfig;
        sourceConfig["signalDataType"] = std::string("CF32");
        sourceConfig["bufferSize"] = std::string("8192");

        REQUIRE(flowgraph->blockCreate("adsb_metadata_src", "signal_generator",
                                       sourceConfig, {}) == Result::SUCCESS);
        Tensor source =
            viewBlock("adsb_metadata_src").outputs.at("signal").tensor;
        REQUIRE(source.setAttribute("sampleRate", F64{2e6}) == Result::SUCCESS);

        TensorMap inputs;
        inputs["signal"].requested("adsb_metadata_src", "signal");
        REQUIRE(flowgraph->blockCreate("adsb_metadata_bad", "adsb", {}, inputs) ==
                Result::SUCCESS);
        REQUIRE(viewBlock("adsb_metadata_bad").state == Block::State::Errored);
        REQUIRE(viewBlock("adsb_metadata_bad").outputs.empty());
    }
}
