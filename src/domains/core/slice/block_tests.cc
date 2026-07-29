#include <catch2/catch_test_macros.hpp>

#include <any>

#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/core/slice/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "jetstream/logger.hh"
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

TEST_CASE_METHOD(FlowgraphFixture, "Slice block propagates module slice validation",
                  "[modules][slice][block][validation]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("slice_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("slice_bad_src", "window");

    Blocks::Slice config;
    config.slice = "[::0]";
    REQUIRE(flowgraph->blockCreate("slice_bad", config, inputs) == Result::SUCCESS);
    const auto block = viewBlock("slice_bad");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.outputs.empty());
    REQUIRE(block.diagnostic.find("[MODULE_SLICE]") != std::string::npos);
}

TEST_CASE_METHOD(FlowgraphFixture, "Slice block preserves topology after rejected update",
                 "[modules][slice][block][reconfigure][validation]") {
    Blocks::Window source;
    source.size = 16;
    REQUIRE(flowgraph->blockCreate("slice_update_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("slice_update_src", "window");

    Blocks::Slice config;
    config.slice = "[0:16:2]";
    config.contiguous = true;
    REQUIRE(flowgraph->blockCreate("slice_update", config, inputs) == Result::SUCCESS);

    const auto original = viewBlock("slice_update").outputs.at("buffer").tensor;
    REQUIRE(original.shape() == Shape{8});
    REQUIRE(original.contiguous());

    Parser::Map rejected;
    rejected["slice"] = std::string("[..., ...]");
    REQUIRE(flowgraph->blockReconfigure("slice_update", rejected) == Result::ERROR);

    const auto block = viewBlock("slice_update");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.at("buffer").tensor.id() == original.id());
    REQUIRE(block.outputs.at("buffer").tensor.shape() == Shape{8});
    REQUIRE(block.outputs.at("buffer").tensor.contiguous());

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("slice_update", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<std::string>(saved.at("slice")) == config.slice);
    REQUIRE(std::any_cast<bool>(saved.at("contiguous")) == config.contiguous);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Slice block changes a rank-three index without diagnostics",
                 "[modules][slice][block][reconfigure]") {
    Blocks::OnesTensor source;
    source.shape = {8, 2, 2024};
    REQUIRE(flowgraph->blockCreate("slice_rank3_src", source, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("slice_rank3_src", "buffer");

    Blocks::Slice config;
    config.slice = "[:, 0, :]";
    config.contiguous = true;
    REQUIRE(flowgraph->blockCreate("slice_rank3", config, inputs) ==
            Result::SUCCESS);

    JST_LOG_LAST_ERROR().clear();
    Parser::Map update;
    update["slice"] = std::string("[:, 1, :]");
    REQUIRE(flowgraph->blockReconfigure("slice_rank3", update) ==
            Result::SUCCESS);
    REQUIRE(JST_LOG_LAST_ERROR().empty());

    const auto output = viewBlock("slice_rank3").outputs.at("buffer").tensor;
    REQUIRE(output.shape() == Shape{8, 2024});
    REQUIRE(output.contiguous());
}
