#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/dsp/decimator/block.hh"

using namespace Jetstream;

TEST_CASE("Decimator axis defaults to the last dimension",
          "[modules][dsp][decimator][config]") {
    REQUIRE(Blocks::Decimator{}.axis == -1);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Decimator block reduces axis by ratio",
                 "[modules][dsp][decimator][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("256");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    Parser::Map decimatorConfig;
    decimatorConfig["axis"] = std::string("0");
    decimatorConfig["ratio"] = std::string("4");

    TensorMap inputs;
    inputs["buffer"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("decimator", "decimator", decimatorConfig, inputs) ==
            Result::SUCCESS);
    const auto block = viewBlock("decimator");
    REQUIRE(block.state == Block::State::Created);

    const auto axis = std::find_if(block.interfaceConfigs.begin(),
                                   block.interfaceConfigs.end(),
                                   [](const auto& entry) { return entry.name == "axis"; });
    REQUIRE(axis != block.interfaceConfigs.end());
    REQUIRE(axis->format == "int:");

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("decimator", saved) == Result::SUCCESS);
    REQUIRE(saved.at("axis").type() == typeid(I64));
    REQUIRE(std::any_cast<I64>(saved.at("axis")) == 0);

    const Tensor out = block.outputs.at("buffer").tensor;
    REQUIRE(out.shape(0) == 64);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Decimator block rejects zero ratio",
                 "[modules][dsp][decimator][block][validation]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    Parser::Map decimatorConfig;
    decimatorConfig["axis"] = std::string("0");
    decimatorConfig["ratio"] = std::string("0");

    TensorMap inputs;
    inputs["buffer"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("decimator_bad", "decimator", decimatorConfig, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("decimator_bad").state == Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Decimator block reduces a multidimensional negative axis",
                 "[modules][dsp][decimator][block][axis]") {
    Blocks::OnesTensor source;
    source.shape = {4, 3};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);

    Blocks::Decimator config;
    config.axis = -2;
    config.ratio = 2;

    TensorMap inputs;
    inputs["buffer"].requested("src", "buffer");
    REQUIRE(flowgraph->blockCreate("decimator", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("decimator").state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor out = viewBlock("decimator").outputs.at("buffer").tensor;
    REQUIRE(out.shape() == Shape{2, 3});
    for (U64 index = 0; index < out.size(); ++index) {
        REQUIRE(out.data<CF32>()[index] == CF32(2.0f, 0.0f));
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Decimator block rejects a too-negative axis",
                 "[modules][dsp][decimator][block][axis][validation]") {
    Blocks::OnesTensor source;
    source.shape = {4, 3};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);

    Blocks::Decimator config;
    config.axis = -3;
    config.ratio = 2;

    TensorMap inputs;
    inputs["buffer"].requested("src", "buffer");
    REQUIRE(flowgraph->blockCreate("decimator_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("decimator_bad").state == Block::State::Errored);
}
