#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/core/duplicate/block.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/core/remove_indices/block.hh"
#include "jetstream/registry.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Remove Indices block creates and serializes its configuration",
                 "[modules][remove_indices][block]") {
    Blocks::OnesTensor source;
    source.shape = {28, 3, 5, 2};
    REQUIRE(flowgraph->blockCreate("source", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("source", "buffer");
    Blocks::RemoveIndices config;
    config.axis = 0;
    config.indices = {2, 7, 19};
    REQUIRE(flowgraph->blockCreate("remove", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("remove").state == Block::State::Created);
    REQUIRE(viewBlock("remove").outputs.at("buffer").tensor.shape() == Shape{25, 3, 5, 2});
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(viewBlock("remove").outputs.at("buffer").tensor.at<F32>(24, 2, 4, 1) == 1.0f);

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("remove", saved) == Result::SUCCESS);
    Blocks::RemoveIndices restored;
    REQUIRE(restored.deserialize(saved) == Result::SUCCESS);
    REQUIRE(restored.axis == config.axis);
    REQUIRE(restored.indices == config.indices);
}

TEST_CASE_METHOD(FlowgraphFixture, "Remove Indices block updates downstream shapes and recovers invalid edits",
                 "[modules][remove_indices][block][reconfigure]") {
    Blocks::OnesTensor source;
    source.shape = {4, 5};
    REQUIRE(flowgraph->blockCreate("source", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("source", "buffer");
    REQUIRE(flowgraph->blockCreate("remove", "remove_indices", {}, inputs) == Result::SUCCESS);

    TensorMap downstreamInputs;
    downstreamInputs["buffer"].requested("remove", "buffer");
    REQUIRE(flowgraph->blockCreate("downstream", Blocks::Duplicate{}, downstreamInputs) == Result::SUCCESS);
    REQUIRE(viewBlock("downstream").outputs.at("buffer").tensor.shape() == Shape{4, 5});

    REQUIRE(flowgraph->blockReconfigure("remove", {{"indices", std::vector<U64>{1, 3}}}) == Result::SUCCESS);
    REQUIRE(viewBlock("downstream").outputs.at("buffer").tensor.shape() == Shape{4, 3});
    REQUIRE(flowgraph->blockReconfigure("remove", {{"axis", I64{0}}}) == Result::SUCCESS);
    REQUIRE(viewBlock("downstream").outputs.at("buffer").tensor.shape() == Shape{2, 5});
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    REQUIRE(flowgraph->blockReconfigure("remove", {{"indices", std::vector<U64>{4}}}) == Result::SUCCESS);
    REQUIRE(viewBlock("remove").state == Block::State::Errored);
    REQUIRE(viewBlock("remove").outputs.empty());
    REQUIRE(viewBlock("downstream").state == Block::State::Incomplete);

    REQUIRE(flowgraph->blockReconfigure("remove", {{"indices", std::vector<U64>{0, 2, 3}}}) == Result::SUCCESS);
    REQUIRE(viewBlock("remove").state == Block::State::Created);
    REQUIRE(viewBlock("downstream").state == Block::State::Created);
    REQUIRE(viewBlock("downstream").outputs.at("buffer").tensor.shape() == Shape{1, 5});
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(viewBlock("downstream").outputs.at("buffer").tensor.at<F32>(0, 4) == 1.0f);

    REQUIRE(flowgraph->blockDisconnect("remove", "buffer") == Result::SUCCESS);
    REQUIRE(viewBlock("remove").state == Block::State::Incomplete);
    REQUIRE(flowgraph->blockConnect("remove", "buffer", "source", "buffer") == Result::SUCCESS);
    REQUIRE(viewBlock("remove").state == Block::State::Created);
    REQUIRE(viewBlock("downstream").state == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture, "Remove Indices block computes and reconfigures on CUDA",
                 "[modules][remove_indices][block][CUDA]") {
    if (Registry::ListAvailableModules("remove_indices", DeviceType::CUDA).empty() ||
        Registry::ListAvailableModules("ones_tensor", DeviceType::CUDA).empty() ||
        Registry::ListAvailableModules("duplicate", DeviceType::CUDA).empty()) {
        SKIP("Required CUDA modules are unavailable.");
    }

    Blocks::OnesTensor source;
    source.shape = {28, 3, 5, 2};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("source", source, {}, DeviceType::CUDA) == Result::SUCCESS);

    Blocks::RemoveIndices config;
    config.axis = 0;
    config.indices = {2, 7, 19};
    TensorMap inputs;
    inputs["buffer"].requested("source", "buffer");
    REQUIRE(flowgraph->blockCreate("remove", config, inputs, DeviceType::CUDA) == Result::SUCCESS);
    const auto output = viewBlock("remove").outputs.at("buffer").tensor;
    REQUIRE(output.device() == DeviceType::CUDA);
    REQUIRE(output.nativeDevice() == DeviceType::CUDA);
    REQUIRE(output.shape() == Shape{25, 3, 5, 2});
    REQUIRE(output.contiguous());

    // Read back only for verification; the source and selection use native GPU memory.
    TensorMap readbackInputs;
    readbackInputs["buffer"].requested("remove", "buffer");
    REQUIRE(flowgraph->blockCreate("readback", Blocks::Duplicate{}, readbackInputs,
                                   DeviceType::CUDA) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(viewBlock("readback").outputs.at("buffer").tensor.at<CF32>(24, 2, 4, 1) ==
            CF32{1.0f, 0.0f});

    REQUIRE(flowgraph->blockReconfigure("remove", {
        {"axis", I64{1}}, {"indices", std::vector<U64>{1}},
    }) == Result::SUCCESS);
    const auto updated = viewBlock("remove").outputs.at("buffer").tensor;
    REQUIRE(updated.device() == DeviceType::CUDA);
    REQUIRE(updated.nativeDevice() == DeviceType::CUDA);
    REQUIRE(updated.shape() == Shape{28, 2, 5, 2});
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(viewBlock("readback").outputs.at("buffer").tensor.at<CF32>(27, 1, 4, 1) ==
            CF32{1.0f, 0.0f});
}
