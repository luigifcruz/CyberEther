#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <blueprint/gain/block.hh>
#include <jetstream/domains/core/ones_tensor/block.hh>
#include <jetstream/flowgraph.hh>
#include <jetstream/flowgraph_view.hh>

using namespace Jetstream;

TEST_CASE("Blueprint gain block creates and computes",
          "[blueprint][gain][block]") {
    Flowgraph flowgraph;
    REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    Blocks::OnesTensor source;
    source.shape = {4};
    REQUIRE(flowgraph.blockCreate("source", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("source", "buffer");

    Blocks::BlueprintGain config;
    config.gain = 3.0f;
    REQUIRE(flowgraph.blockCreate("gain", config, inputs) == Result::SUCCESS);

    Flowgraph::View::BlockData gain;
    REQUIRE(flowgraph.view().block("gain", gain) == Result::SUCCESS);
    REQUIRE(gain.state == Block::State::Created);
    REQUIRE(gain.outputs.count("signal") == 1);

    REQUIRE(flowgraph.compute() == Result::SUCCESS);

    REQUIRE(flowgraph.view().block("gain", gain) == Result::SUCCESS);
    const Tensor output = gain.outputs.at("signal").tensor;
    REQUIRE(output.shape() == Shape{4});
    REQUIRE(output.dtype() == DataType::F32);
    for (U64 index = 0; index < output.size(); ++index) {
        REQUIRE_THAT(output.at<F32>(index),
                     Catch::Matchers::WithinAbs(3.0f, 1e-6f));
    }

    REQUIRE(flowgraph.blockDestroy("gain", false) == Result::SUCCESS);
    REQUIRE(flowgraph.blockDestroy("source", false) == Result::SUCCESS);
    REQUIRE(flowgraph.destroy() == Result::SUCCESS);
}

TEST_CASE("Blueprint gain block reconnects its input",
          "[blueprint][gain][block][lifecycle]") {
    Flowgraph flowgraph;
    REQUIRE(flowgraph.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);

    Blocks::OnesTensor source;
    REQUIRE(flowgraph.blockCreate("source", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("source", "buffer");
    REQUIRE(flowgraph.blockCreate("gain", "blueprint_gain", {}, inputs) ==
            Result::SUCCESS);

    REQUIRE(flowgraph.blockDisconnect("gain", "signal") == Result::SUCCESS);

    Flowgraph::View::BlockData gain;
    REQUIRE(flowgraph.view().block("gain", gain) == Result::SUCCESS);
    REQUIRE(gain.state == Block::State::Incomplete);

    REQUIRE(flowgraph.blockConnect("gain", "signal", "source", "buffer") ==
            Result::SUCCESS);
    REQUIRE(flowgraph.view().block("gain", gain) == Result::SUCCESS);
    REQUIRE(gain.state == Block::State::Created);

    REQUIRE(flowgraph.blockDestroy("gain", false) == Result::SUCCESS);
    REQUIRE(flowgraph.blockDestroy("source", false) == Result::SUCCESS);
    REQUIRE(flowgraph.destroy() == Result::SUCCESS);
}
