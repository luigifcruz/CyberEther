#include <catch2/catch_test_macros.hpp>

#include <any>
#include <string>

#include "jetstream/domains/core/ones_tensor/block.hh"
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
    config.interpolate = false;

    REQUIRE(flowgraph->blockCreate("waterfall", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("waterfall").state == Block::State::Created);
    REQUIRE(viewBlock("waterfall").outputs.empty());

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
    config["interpolate"] = std::string("false");
    REQUIRE(flowgraph->blockReconfigure("waterfall", config) == Result::SUCCESS);
    REQUIRE(viewBlock("waterfall").state == Block::State::Created);

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
    Blocks::OnesTensor source;
    source.shape = {8};
    source.dataType = "F64";
    REQUIRE(flowgraph->blockCreate("waterfall_dtype_src", source, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("waterfall_dtype_src", "buffer");

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
    config.interpolate = false;
    REQUIRE(flowgraph->blockCreate("waterfall_update", config, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    Parser::Map update;
    update["height"] = U64{0};
    update["interpolate"] = true;
    REQUIRE(flowgraph->blockReconfigure("waterfall_update", update) == Result::SUCCESS);
    REQUIRE(viewBlock("waterfall_update").state == Block::State::Errored);

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("waterfall_update", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(saved.at("height")) == 0);
    REQUIRE(std::any_cast<bool>(saved.at("interpolate")));

    Parser::Map recovery;
    recovery["height"] = config.height;
    REQUIRE(flowgraph->blockReconfigure("waterfall_update", recovery) == Result::SUCCESS);
    REQUIRE(viewBlock("waterfall_update").state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
}
