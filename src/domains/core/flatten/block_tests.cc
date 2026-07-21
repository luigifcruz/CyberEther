#include <catch2/catch_test_macros.hpp>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/core/flatten/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "jetstream/registry.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Flatten block creates",
                 "[modules][flatten][block]") {
    Blocks::Window source;
    source.size = 8;
    REQUIRE(flowgraph->blockCreate("flatten_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("flatten_src", "window");

    Blocks::Flatten config;
    REQUIRE(flowgraph->blockCreate("flatten_block", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("flatten_block").state == Block::State::Created);
    REQUIRE(viewBlock("flatten_block").outputs.count("buffer") == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "Flatten block keeps contiguous copies on CUDA",
                 "[modules][flatten][block][CUDA]") {
    if (Registry::ListAvailableModules("window", DeviceType::CUDA).empty() ||
        Registry::ListAvailableModules("duplicate", DeviceType::CUDA).empty() ||
        Registry::ListAvailableModules("flatten", DeviceType::CUDA).empty()) {
        SUCCEED("Required CUDA modules are unavailable in this build.");
        return;
    }

    Blocks::Window source;
    source.size = 8;
    REQUIRE(flowgraph->blockCreate("flatten_cuda_src", source, {}, DeviceType::CUDA) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("flatten_cuda_src", "window");

    Blocks::Flatten config;
    config.contiguous = true;
    REQUIRE(flowgraph->blockCreate("flatten_cuda", config, inputs, DeviceType::CUDA) ==
            Result::SUCCESS);

    const auto block = viewBlock("flatten_cuda");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.at("buffer").tensor.device() == DeviceType::CUDA);
}
