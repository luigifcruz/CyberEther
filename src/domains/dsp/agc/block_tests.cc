#include <catch2/catch_test_macros.hpp>

#include <any>
#include <string>

#include "jetstream/domains/dsp/agc/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "AGC block wires signal input and output",
                 "[modules][dsp][agc][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("128");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    Blocks::Agc config;
    config.tileSize = 64;
    config.reference = 0.5f;
    config.epsilon = 1e-9f;
    config.minGain = 0.1f;
    config.maxGain = 10.0f;
    config.maxGainChange = 2.0f;
    REQUIRE(flowgraph->blockCreate("agc", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("agc").state == Block::State::Created);

    const auto block = viewBlock("agc");
    const Tensor out = block.outputs.at("signal").tensor;
    REQUIRE(out.dtype() == DataType::CF32);
    REQUIRE(out.rank() == 1);
    REQUIRE(out.shape(0) == 128);

    REQUIRE(block.interfaceConfigs.size() == 6);
    REQUIRE(block.interfaceConfigs.at(0).name == "tileSize");
    REQUIRE(block.interfaceConfigs.at(1).name == "reference");
    REQUIRE(block.interfaceConfigs.at(2).name == "epsilon");
    REQUIRE(block.interfaceConfigs.at(3).name == "minGain");
    REQUIRE(block.interfaceConfigs.at(4).name == "maxGain");
    REQUIRE(block.interfaceConfigs.at(5).name == "maxGainChange");

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("agc", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(saved.at("tileSize")) == config.tileSize);
    REQUIRE(std::any_cast<F32>(saved.at("reference")) == config.reference);
    REQUIRE(std::any_cast<F32>(saved.at("epsilon")) == config.epsilon);
    REQUIRE(std::any_cast<F32>(saved.at("minGain")) == config.minGain);
    REQUIRE(std::any_cast<F32>(saved.at("maxGain")) == config.maxGain);
    REQUIRE(std::any_cast<F32>(saved.at("maxGainChange")) ==
            config.maxGainChange);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "AGC block is incomplete without required input",
                 "[modules][dsp][agc][block][validation]") {
    auto result = flowgraph->blockCreate("agc_incomplete", "agc", {}, {});
    REQUIRE((result == Result::SUCCESS || result == Result::INCOMPLETE));
    REQUIRE(viewBlock("agc_incomplete").state ==
            Block::State::Incomplete);
}
