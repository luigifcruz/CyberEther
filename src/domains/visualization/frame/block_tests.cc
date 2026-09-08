#include <catch2/catch_test_macros.hpp>

#include <algorithm>
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

TEST_CASE_METHOD(FlowgraphFixture,
                 "Frame block delegates dtype validation to its module",
                 "[modules][frame][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {16, 32};
    source.dataType = "F64";
    REQUIRE(flowgraph->blockCreate("frame_dtype_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["frame"].requested("frame_dtype_src", "buffer");

    REQUIRE(flowgraph->blockCreate("frame_dtype", Blocks::Frame{}, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("frame_dtype").state == Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Frame block exposes LUT configuration only for scalar frames",
                 "[modules][frame][block][interface]") {
    Blocks::OnesTensor scalarSource;
    scalarSource.shape = {16, 32};
    scalarSource.dataType = "F32";
    REQUIRE(flowgraph->blockCreate("frame_scalar_src", scalarSource, {}) ==
            Result::SUCCESS);

    TensorMap scalarInputs;
    scalarInputs["frame"].requested("frame_scalar_src", "buffer");
    REQUIRE(flowgraph->blockCreate("frame_scalar", Blocks::Frame{}, scalarInputs) ==
            Result::SUCCESS);

    const auto scalar = viewBlock("frame_scalar");
    const auto scalarLut = std::find_if(
        scalar.interfaceConfigs.begin(),
        scalar.interfaceConfigs.end(),
        [](const auto& field) { return field.name == "lut"; });
    REQUIRE(scalarLut != scalar.interfaceConfigs.end());

    Blocks::OnesTensor colorSource;
    colorSource.shape = {16, 32, 3};
    colorSource.dataType = "F32";
    REQUIRE(flowgraph->blockCreate("frame_color_src", colorSource, {}) ==
            Result::SUCCESS);

    TensorMap colorInputs;
    colorInputs["frame"].requested("frame_color_src", "buffer");
    REQUIRE(flowgraph->blockCreate("frame_color", Blocks::Frame{}, colorInputs) ==
            Result::SUCCESS);

    const auto color = viewBlock("frame_color");
    const auto colorLut = std::find_if(
        color.interfaceConfigs.begin(),
        color.interfaceConfigs.end(),
        [](const auto& field) { return field.name == "lut"; });
    REQUIRE(colorLut == color.interfaceConfigs.end());
}
