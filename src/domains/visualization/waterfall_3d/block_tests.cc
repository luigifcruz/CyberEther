#include <catch2/catch_test_macros.hpp>

#include <any>
#include <algorithm>
#include <string>

#include "jetstream/domains/dsp/signal_generator/block.hh"
#include "jetstream/domains/visualization/waterfall_3d/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "3D Waterfall block create and lifecycle",
                 "[modules][waterfall_3d][block]") {
    Blocks::SignalGenerator sourceConfig;
    sourceConfig.signalDataType = "F32";
    sourceConfig.bufferSize = 64;

    REQUIRE(flowgraph->blockCreate("src", sourceConfig, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    Blocks::Waterfall3D config;
    config.height = 64;
    config.xLabel = "Frequency";
    config.timeLabel = "History";
    config.amplitudeLabel = "Power";

    REQUIRE(flowgraph->blockCreate("surface", config, inputs) == Result::SUCCESS);
    const auto block = viewBlock("surface");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.empty());
    REQUIRE(std::any_cast<U64>(block.config.at("averaging")) == 1);
    const auto averaging = std::find_if(block.interfaceConfigs.begin(),
                                        block.interfaceConfigs.end(),
                                        [](const auto& entry) { return entry.name == "averaging"; });
    REQUIRE(averaging != block.interfaceConfigs.end());
    REQUIRE(averaging->format == Parser::Map{
        {"type", "range"}, {"min", 1.0f}, {"max", 64.0f}, {"value_type", "uint"},
    });
    REQUIRE(std::any_of(block.interfaceConfigs.begin(), block.interfaceConfigs.end(),
                        [](const auto& entry) { return entry.name == "height"; }));
    REQUIRE(std::any_cast<std::string>(block.config.at("xLabel")) == "Frequency");
    REQUIRE(std::any_cast<std::string>(block.config.at("timeLabel")) == "History");
    REQUIRE(std::any_cast<std::string>(block.config.at("amplitudeLabel")) == "Power");
    for (const auto& entry : block.interfaceConfigs) {
        REQUIRE(entry.name != "xLabel");
        REQUIRE(entry.name != "timeLabel");
        REQUIRE(entry.name != "amplitudeLabel");
    }

    auto result = flowgraph->blockDisconnect("surface", "signal");
    REQUIRE((result == Result::SUCCESS || result == Result::INCOMPLETE));
    REQUIRE(viewBlock("surface").state == Block::State::Incomplete);

    REQUIRE(flowgraph->blockConnect("surface", "signal", "src", "signal") ==
            Result::SUCCESS);
    REQUIRE(viewBlock("surface").state == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "3D Waterfall block reconfigure and validation",
                 "[modules][waterfall_3d][block][validation]") {
    Blocks::SignalGenerator sourceConfig;
    sourceConfig.signalDataType = "F32";
    sourceConfig.bufferSize = 64;

    REQUIRE(flowgraph->blockCreate("src", sourceConfig, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("surface", Blocks::Waterfall3D(), inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    Parser::Map config;
    config["height"] = std::string("128");
    config["xLabel"] = std::string("Channel");
    config["timeLabel"] = std::string("Elapsed Time");
    REQUIRE(flowgraph->blockReconfigure("surface", config) == Result::SUCCESS);
    const auto reconfigured = viewBlock("surface");
    REQUIRE(reconfigured.state == Block::State::Created);
    REQUIRE(std::any_cast<U64>(reconfigured.config.at("height")) == 128);
    REQUIRE(std::any_cast<std::string>(reconfigured.config.at("xLabel")) == "Channel");
    REQUIRE(std::any_cast<std::string>(reconfigured.config.at("timeLabel")) == "Elapsed Time");
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    Parser::Map invalid;
    SECTION("height below two rows") {
        invalid["height"] = U64{1};
    }
    SECTION("zero averaging") {
        invalid["averaging"] = U64{0};
    }
    REQUIRE(flowgraph->blockReconfigure("surface", invalid) == Result::SUCCESS);
    REQUIRE(viewBlock("surface").state == Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "3D Waterfall block delegates dtype validation to its module",
                 "[modules][waterfall_3d][block][validation]") {
    Blocks::SignalGenerator source;
    source.bufferSize = 8;
    source.signalDataType = "CF32";
    REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("surface", Blocks::Waterfall3D{}, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("surface").state == Block::State::Errored);
}
