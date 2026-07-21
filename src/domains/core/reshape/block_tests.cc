#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "jetstream/domains/core/reshape/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "jetstream/registry.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Reshape block creates with target shape",
                 "[modules][reshape][block]") {
    Blocks::Window source;
    source.size = 8;
    REQUIRE(flowgraph->blockCreate("reshape_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("reshape_src", "window");

    Blocks::Reshape config;
    config.shape = "[2, 4]";
    config.contiguous = true;
    REQUIRE(flowgraph->blockCreate("reshape_block", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("reshape_block").state == Block::State::Created);
    REQUIRE(viewBlock("reshape_block").outputs.count("buffer") == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "Reshape block keeps contiguous copies on CUDA",
                 "[modules][reshape][block][CUDA]") {
    if (Registry::ListAvailableModules("window", DeviceType::CUDA).empty() ||
        Registry::ListAvailableModules("duplicate", DeviceType::CUDA).empty() ||
        Registry::ListAvailableModules("reshape", DeviceType::CUDA).empty()) {
        SUCCEED("Required CUDA modules are unavailable in this build.");
        return;
    }

    Blocks::Window source;
    source.size = 8;
    REQUIRE(flowgraph->blockCreate("reshape_cuda_src", source, {}, DeviceType::CUDA) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("reshape_cuda_src", "window");

    Blocks::Reshape config;
    config.shape = "[2, 4]";
    config.contiguous = true;
    REQUIRE(flowgraph->blockCreate("reshape_cuda", config, inputs, DeviceType::CUDA) ==
            Result::SUCCESS);

    const auto block = viewBlock("reshape_cuda");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.at("buffer").tensor.device() == DeviceType::CUDA);
}

TEST_CASE_METHOD(FlowgraphFixture, "Reshape block recovers from invalid target shape",
                  "[modules][reshape][block][validation]") {
    Blocks::Window source;
    source.size = 8;
    REQUIRE(flowgraph->blockCreate("reshape_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("reshape_bad_src", "window");

    Blocks::Reshape config;
    config.shape = "[7]";
    REQUIRE(flowgraph->blockCreate("reshape_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("reshape_bad").state == Block::State::Errored);

    config.shape = "[8]";
    Parser::Map updated;
    REQUIRE(config.serialize(updated) == Result::SUCCESS);
    REQUIRE(flowgraph->blockReconfigure("reshape_bad", updated) == Result::SUCCESS);
    REQUIRE(viewBlock("reshape_bad").state == Block::State::Created);
    REQUIRE(viewBlock("reshape_bad").outputs.count("buffer") == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "Reshape settles when its input is static",
                 "[modules][reshape][block][static]") {
    Blocks::Window source;
    source.size = 8;
    REQUIRE(flowgraph->blockCreate("reshape_static_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("reshape_static_src", "window");

    Blocks::Reshape config;
    config.shape = "[2, 4]";
    config.contiguous = false;
    REQUIRE(flowgraph->blockCreate("reshape_static", config, inputs) == Result::SUCCESS);

    for (U64 cycle = 0; cycle < 3; ++cycle) {
        REQUIRE(flowgraph->compute() == Result::SUCCESS);
    }

    std::vector<Flowgraph::View::MetricEntry> sourceMetrics;
    REQUIRE(flowgraph->view().metrics("reshape_static_src", sourceMetrics) == Result::SUCCESS);
    REQUIRE(sourceMetrics.size() == 1);
    const auto* sourceTiming = std::any_cast<Module::Timing>(&sourceMetrics.front().value);
    REQUIRE(sourceTiming != nullptr);
    REQUIRE(sourceTiming->cycles == 1);
    REQUIRE(sourceTiming->computeTime == 0.0f);

    std::vector<Flowgraph::View::MetricEntry> reshapeMetrics;
    REQUIRE(flowgraph->view().metrics("reshape_static", reshapeMetrics) == Result::SUCCESS);
    REQUIRE(reshapeMetrics.size() == 1);
    REQUIRE(reshapeMetrics.front().name == "runtime:reshape");
    const auto* reshapeTiming = std::any_cast<Module::Timing>(&reshapeMetrics.front().value);
    REQUIRE(reshapeTiming != nullptr);
    REQUIRE(reshapeTiming->cycles == 1);
    REQUIRE(reshapeTiming->computeTime == 0.0f);
}
