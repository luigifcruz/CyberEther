#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/core/cast/block.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Cast block bypasses matching source type",
                  "[modules][cast][block]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("cast_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("cast_src", "window");

    Blocks::Cast config;
    config.outputType = "CF32";
    REQUIRE(flowgraph->blockCreate("cast_block", config, inputs) == Result::SUCCESS);

    const auto castBlock = viewBlock("cast_block");
    REQUIRE(castBlock.state == Block::State::Created);
    REQUIRE(castBlock.outputs.at("buffer").tensor.dtype() == DataType::CF32);
}

TEST_CASE_METHOD(FlowgraphFixture, "Cast block delegates invalid output type",
                  "[modules][cast][block][validation]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("cast_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("cast_bad_src", "window");

    Blocks::Cast config;
    config.outputType = "INVALID";
    REQUIRE(flowgraph->blockCreate("cast_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("cast_bad").state == Block::State::Errored);
    REQUIRE(viewBlock("cast_bad").outputs.empty());
}

TEST_CASE_METHOD(FlowgraphFixture, "Cast block delegates unsupported conversion pair",
                 "[modules][cast][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {4};
    source.dataType = "F64";
    REQUIRE(flowgraph->blockCreate("cast_pair_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("cast_pair_src", "buffer");

    Blocks::Cast config;
    config.outputType = "CF32";
    REQUIRE(flowgraph->blockCreate("cast_pair", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("cast_pair").state == Block::State::Errored);
    REQUIRE(viewBlock("cast_pair").outputs.empty());
}

TEST_CASE_METHOD(FlowgraphFixture, "Cast block preserves invalid output type for recovery",
                 "[modules][cast][block][reconfigure][validation]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("cast_update_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("cast_update_src", "window");

    Blocks::Cast config;
    config.outputType = "CF32";
    REQUIRE(flowgraph->blockCreate("cast_update", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    Parser::Map update;
    update["outputType"] = std::string("F32");
    REQUIRE(flowgraph->blockReconfigure("cast_update", update) == Result::SUCCESS);
    REQUIRE(viewBlock("cast_update").state == Block::State::Errored);
    REQUIRE(viewBlock("cast_update").outputs.empty());

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("cast_update", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<std::string>(saved.at("outputType")) == "F32");

    Parser::Map recovery;
    recovery["outputType"] = config.outputType;
    REQUIRE(flowgraph->blockReconfigure("cast_update", recovery) == Result::SUCCESS);
    REQUIRE(viewBlock("cast_update").state == Block::State::Created);

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    const Tensor output = viewBlock("cast_update").outputs.at("buffer").tensor;
    REQUIRE(output.dtype() == DataType::CF32);
}
