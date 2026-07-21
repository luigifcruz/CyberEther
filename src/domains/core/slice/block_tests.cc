#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/core/slice/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "jetstream/registry.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Slice block creates with contiguous output",
                 "[modules][slice][block]") {
    Blocks::Window source;
    source.size = 16;
    REQUIRE(flowgraph->blockCreate("slice_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("slice_src", "window");

    Blocks::Slice config;
    config.slice = "[0:8]";
    config.contiguous = true;
    REQUIRE(flowgraph->blockCreate("slice_block", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("slice_block").state == Block::State::Created);
    REQUIRE(viewBlock("slice_block").outputs.count("buffer") == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "Slice block keeps contiguous copies on CUDA",
                 "[modules][slice][block][CUDA]") {
    if (Registry::ListAvailableModules("window", DeviceType::CUDA).empty() ||
        Registry::ListAvailableModules("slice", DeviceType::CUDA).empty() ||
        Registry::ListAvailableModules("duplicate", DeviceType::CUDA).empty()) {
        SUCCEED("Required CUDA modules are unavailable in this build.");
        return;
    }

    Blocks::Window source;
    source.size = 16;
    REQUIRE(flowgraph->blockCreate("slice_cuda_src", source, {}, DeviceType::CUDA) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("slice_cuda_src", "window");

    Blocks::Slice config;
    config.slice = "[0:8]";
    config.contiguous = true;
    REQUIRE(flowgraph->blockCreate("slice_cuda", config, inputs, DeviceType::CUDA) ==
            Result::SUCCESS);

    const auto block = viewBlock("slice_cuda");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.at("buffer").tensor.device() == DeviceType::CUDA);
}

TEST_CASE_METHOD(FlowgraphFixture, "Slice block rejects malformed slice string",
                 "[modules][slice][block][validation]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("slice_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("slice_bad_src", "window");

    Blocks::Slice config;
    config.slice = "foo";
    REQUIRE(flowgraph->blockCreate("slice_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("slice_bad").state == Block::State::Errored);
}
