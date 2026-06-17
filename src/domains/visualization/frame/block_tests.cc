#include <catch2/catch_test_macros.hpp>

#include <string>

#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/visualization/frame/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Frame block create and lifecycle",
                 "[modules][frame][block]") {
    Blocks::OnesTensor sourceConfig;
    sourceConfig.shape = {16, 32};
    sourceConfig.dataType = "F32";

    REQUIRE(flowgraph->blockCreate("src", sourceConfig, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["frame"].requested("src", "buffer");

    Blocks::Frame config;
    config.lut = true;

    REQUIRE(flowgraph->blockCreate("frame", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("frame").state == Block::State::Created);
    REQUIRE(viewBlock("frame").outputs.empty());

    auto result = flowgraph->blockDisconnect("frame", "frame");
    REQUIRE((result == Result::SUCCESS || result == Result::INCOMPLETE));
    REQUIRE(viewBlock("frame").state == Block::State::Incomplete);

    REQUIRE(flowgraph->blockConnect("frame", "frame", "src", "buffer") ==
            Result::SUCCESS);
    REQUIRE(viewBlock("frame").state == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Frame block reconfigure and validation",
                 "[modules][frame][block][validation]") {
    Blocks::OnesTensor sourceConfig;
    sourceConfig.shape = {16, 32};
    sourceConfig.dataType = "F32";

    REQUIRE(flowgraph->blockCreate("src", sourceConfig, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["frame"].requested("src", "buffer");

    REQUIRE(flowgraph->blockCreate("frame", Blocks::Frame(), inputs) ==
            Result::SUCCESS);

    Parser::Map config;
    config["lut"] = std::string("true");
    REQUIRE(flowgraph->blockReconfigure("frame", config) == Result::SUCCESS);
    REQUIRE(viewBlock("frame").state == Block::State::Created);

    Blocks::OnesTensor invalidSource;
    invalidSource.shape = {32};
    invalidSource.dataType = "F32";
    REQUIRE(flowgraph->blockCreate("invalid_src", invalidSource, {}) ==
            Result::SUCCESS);

    TensorMap invalidInputs;
    invalidInputs["frame"].requested("invalid_src", "buffer");

    REQUIRE(flowgraph->blockCreate("frame_invalid", Blocks::Frame(), invalidInputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("frame_invalid").state == Block::State::Errored);
}
