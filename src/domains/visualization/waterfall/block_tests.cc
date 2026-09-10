#include <catch2/catch_test_macros.hpp>

#include <any>
#include <algorithm>
#include <string>

#include "jetstream/domains/dsp/signal_generator/block.hh"
#include "jetstream/domains/visualization/waterfall/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Waterfall block create and lifecycle",
                 "[modules][waterfall][block]") {
    Blocks::SignalGenerator sourceConfig;
    sourceConfig.signalDataType = "F32";
    sourceConfig.bufferSize = 64;

    REQUIRE(flowgraph->blockCreate("src", sourceConfig, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    Blocks::Waterfall config;
    config.height = 64;
    config.xLabel = "Frequency";
    config.yLabel = "History";

    REQUIRE(flowgraph->blockCreate("waterfall", config, inputs) == Result::SUCCESS);
    const auto block = viewBlock("waterfall");
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
    REQUIRE(std::any_cast<std::string>(block.config.at("xLabel")) == "Frequency");
    REQUIRE(std::any_cast<std::string>(block.config.at("yLabel")) == "History");
    for (const auto& entry : block.interfaceConfigs) {
        REQUIRE(entry.name != "xLabel");
        REQUIRE(entry.name != "yLabel");
    }

    auto result = flowgraph->blockDisconnect("waterfall", "signal");
    REQUIRE((result == Result::SUCCESS || result == Result::INCOMPLETE));
    REQUIRE(viewBlock("waterfall").state ==
            Block::State::Incomplete);

    REQUIRE(flowgraph->blockConnect("waterfall", "signal", "src", "signal") ==
            Result::SUCCESS);
    REQUIRE(viewBlock("waterfall").state == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Waterfall block reconfigure and validation",
                 "[modules][waterfall][block][validation]") {
    Blocks::SignalGenerator sourceConfig;
    sourceConfig.signalDataType = "F32";
    sourceConfig.bufferSize = 64;

    REQUIRE(flowgraph->blockCreate("src", sourceConfig, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("waterfall", Blocks::Waterfall(), inputs) ==
            Result::SUCCESS);

    Parser::Map config;
    config["height"] = std::string("128");
    config["xLabel"] = std::string("Channel");
    config["yLabel"] = std::string("Elapsed Time");
    REQUIRE(flowgraph->blockReconfigure("waterfall", config) == Result::SUCCESS);
    const auto reconfigured = viewBlock("waterfall");
    REQUIRE(reconfigured.state == Block::State::Created);
    REQUIRE(std::any_cast<std::string>(reconfigured.config.at("xLabel")) ==
            "Channel");
    REQUIRE(std::any_cast<std::string>(reconfigured.config.at("yLabel")) ==
            "Elapsed Time");

    Blocks::Waterfall invalid;
    invalid.height = 0;
    REQUIRE(flowgraph->blockCreate("waterfall_invalid", invalid, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("waterfall_invalid").state ==
            Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Waterfall block delegates dtype validation to its module",
                 "[modules][waterfall][block][validation]") {
    Blocks::SignalGenerator source;
    source.bufferSize = 8;
    source.signalDataType = "CF32";
    REQUIRE(flowgraph->blockCreate("waterfall_dtype_src", source, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("waterfall_dtype_src", "signal");

    REQUIRE(flowgraph->blockCreate("waterfall_dtype", Blocks::Waterfall{}, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("waterfall_dtype").state == Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Waterfall block preserves invalid config for recovery",
                 "[modules][waterfall][block][reconfigure][validation]") {
    Blocks::SignalGenerator source;
    source.signalDataType = "F32";
    source.bufferSize = 64;
    REQUIRE(flowgraph->blockCreate("waterfall_update_src", source, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("waterfall_update_src", "signal");

    Blocks::Waterfall config;
    config.height = 128;
    REQUIRE(flowgraph->blockCreate("waterfall_update", config, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    Parser::Map update;
    SECTION("invalid height") {
        update["height"] = U64{0};
    }
    SECTION("invalid averaging") {
        update["averaging"] = U64{0};
    }
    REQUIRE(flowgraph->blockReconfigure("waterfall_update", update) == Result::SUCCESS);
    REQUIRE(viewBlock("waterfall_update").state == Block::State::Errored);

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("waterfall_update", saved) == Result::SUCCESS);
    for (const auto& [key, value] : update) {
        REQUIRE(std::any_cast<U64>(saved.at(key)) == std::any_cast<U64>(value));
    }

    Parser::Map recovery;
    recovery["height"] = config.height;
    recovery["averaging"] = U64{4};
    REQUIRE(flowgraph->blockReconfigure("waterfall_update", recovery) == Result::SUCCESS);
    REQUIRE(viewBlock("waterfall_update").state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Waterfall averaging changes in place and survives export",
                 "[modules][waterfall][block][averaging][reconfigure]") {
    Blocks::SignalGenerator source;
    source.bufferSize = 8;
    REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);
    TensorMap inputs;
    inputs["signal"].requested("src", "signal");
    Blocks::Waterfall config;
    config.height = 8;
    config.averaging = 2;
    REQUIRE(flowgraph->blockCreate("waterfall", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    const auto surface = viewBlock("waterfall").surfaces.front();

    Parser::Map update;
    update["averaging"] = std::string("16");
    REQUIRE(flowgraph->blockReconfigure("waterfall", update) == Result::SUCCESS);
    REQUIRE(viewBlock("waterfall").surfaces.front() == surface);
    REQUIRE(std::any_cast<U64>(viewBlock("waterfall").config.at("averaging")) == 16);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    std::vector<char> blob;
    REQUIRE(flowgraph->exportToBlob(blob) == Result::SUCCESS);
    Flowgraph restored;
    REQUIRE(restored.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    REQUIRE(restored.importFromBlob(blob) == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(ViewBlock(restored, "waterfall").config.at("averaging")) == 16);
    REQUIRE(restored.compute() == Result::SUCCESS);
    REQUIRE(restored.destroy() == Result::SUCCESS);
}
