#include <catch2/catch_test_macros.hpp>

#include <any>
#include <string>
#include <vector>

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
    REQUIRE(viewBlock("filter").state == Block::State::Created);
    REQUIRE(viewBlock("filter").outputs.contains("buffer"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter block uses heads over center length",
                 "[modules][dsp][filter][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("512");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    Parser::Map config;
    config["heads"] = std::string("1");
    config["center"] = std::string("[0, 400000]");

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("filter_single", "filter", config, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("filter_single").state == Block::State::Created);
    REQUIRE(viewBlock("filter_single").outputs.contains("buffer"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter block heads can shrink stale center vector",
                 "[modules][dsp][filter][block][reconfigure]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("512");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    Parser::Map config;
    config["heads"] = std::string("4");
    config["center"] = std::string("[0, 100000, -100000, 200000]");

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("filter_shrink", "filter", config, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("filter_shrink").state == Block::State::Created);

    Parser::Map update;
    update["sampleRate"] = std::string("2000000");
    update["bandwidth"] = std::string("1000000");
    update["taps"] = std::string("101");
    update["heads"] = std::string("1");
    update["center"] = std::string("[0, 100000, -100000, 200000]");

    REQUIRE(flowgraph->blockReconfigure("filter_shrink", update) == Result::SUCCESS);
    REQUIRE(viewBlock("filter_shrink").state == Block::State::Created);

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("filter_shrink", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(saved.at("heads")) == 1);
    REQUIRE(std::any_cast<std::vector<F32>>(saved.at("center")).size() == 1);
}
