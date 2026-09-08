#include <catch2/catch_test_macros.hpp>

#include <any>
#include <string>

#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/dsp/signal_generator/block.hh"
#include "jetstream/domains/visualization/constellation/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Constellation block wiring and lifecycle",
                 "[modules][constellation][block]") {
    Blocks::SignalGenerator sourceConfig;
    sourceConfig.signalDataType = "CF32";
    sourceConfig.bufferSize = 64;

    REQUIRE(flowgraph->blockCreate("src", sourceConfig, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    Blocks::Constellation constellationConfig;
    constellationConfig.xLabel = "In Phase";
    constellationConfig.yLabel = "Quadrature";
    REQUIRE(flowgraph->blockCreate("constellation", constellationConfig, inputs) ==
            Result::SUCCESS);
    const auto block = viewBlock("constellation");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.empty());
    REQUIRE(std::any_cast<std::string>(block.config.at("xLabel")) == "In Phase");
    REQUIRE(std::any_cast<std::string>(block.config.at("yLabel")) == "Quadrature");
    for (const auto& entry : block.interfaceConfigs) {
        REQUIRE(entry.name != "xLabel");
        REQUIRE(entry.name != "yLabel");
    }

    Parser::Map update;
    update["xLabel"] = std::string("Real");
    update["yLabel"] = std::string("Imaginary");
    REQUIRE(flowgraph->blockReconfigure("constellation", update) == Result::SUCCESS);
    const auto reconfigured = viewBlock("constellation");
    REQUIRE(std::any_cast<std::string>(reconfigured.config.at("xLabel")) == "Real");
    REQUIRE(std::any_cast<std::string>(reconfigured.config.at("yLabel")) ==
            "Imaginary");

    auto result = flowgraph->blockDisconnect("constellation", "signal");
    REQUIRE((result == Result::SUCCESS || result == Result::INCOMPLETE));
    REQUIRE(viewBlock("constellation").state ==
            Block::State::Incomplete);

    REQUIRE(flowgraph->blockConnect("constellation", "signal", "src", "signal") ==
            Result::SUCCESS);
    REQUIRE(viewBlock("constellation").state ==
            Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Constellation block delegates rank validation to its module",
                 "[modules][constellation][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {2, 2, 2};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("constellation_rank_src", source, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("constellation_rank_src", "buffer");

    REQUIRE(flowgraph->blockCreate("constellation_rank", Blocks::Constellation{}, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("constellation_rank").state == Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Constellation block delegates dtype validation to its module",
                 "[modules][constellation][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {8};
    source.dataType = "F32";
    REQUIRE(flowgraph->blockCreate("constellation_dtype_src", source, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("constellation_dtype_src", "buffer");

    REQUIRE(flowgraph->blockCreate("constellation_dtype", Blocks::Constellation{}, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("constellation_dtype").state == Block::State::Errored);
}
