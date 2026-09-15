#include <catch2/catch_test_macros.hpp>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/dsp/rational_resampler/block.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Rational resampler block exposes rate conversion and recovers from edits",
                 "[modules][dsp][rational_resampler][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = U64{1024};
    sourceConfig["sampleRate"] = F32{250000};
    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);
    TensorMap inputs;
    inputs["buffer"].requested("src", "signal");
    Blocks::RationalResampler config;
    config.interpolation = 24;
    config.decimation = 125;
    REQUIRE(flowgraph->blockCreate("resampler", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("resampler").state == Block::State::Created);
    const Tensor output = viewBlock("resampler").outputs.at("buffer").tensor;
    REQUIRE(output.dtype() == DataType::CF32);
    REQUIRE(output.shape() == Shape{197});
    REQUIRE(std::any_cast<F32>(output.attribute("sampleRate")) == 48000.0f);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    Parser::Map update;
    update["interpolation"] = U64{3};
    update["decimation"] = U64{2};
    REQUIRE(flowgraph->blockReconfigure("resampler", update) == Result::SUCCESS);
    REQUIRE(viewBlock("resampler").state == Block::State::Created);
    const Tensor changed = viewBlock("resampler").outputs.at("buffer").tensor;
    REQUIRE(changed.shape() == Shape{1536});
    REQUIRE(changed.id() != output.id());
    REQUIRE(std::any_cast<F32>(changed.attribute("sampleRate")) == 375000.0f);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    update.clear();
    update["decimation"] = U64{0};
    REQUIRE(flowgraph->blockReconfigure("resampler", update) == Result::SUCCESS);
    REQUIRE(viewBlock("resampler").state == Block::State::Errored);
    REQUIRE(viewBlock("resampler").outputs.empty());
    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("resampler", saved) == Result::SUCCESS);
    Blocks::RationalResampler restored;
    REQUIRE(restored.deserialize(saved) == Result::SUCCESS);
    REQUIRE(restored.interpolation == 3);
    REQUIRE(restored.decimation == 0);
    update["decimation"] = U64{2};
    REQUIRE(flowgraph->blockReconfigure("resampler", update) == Result::SUCCESS);
    REQUIRE(viewBlock("resampler").state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
}
