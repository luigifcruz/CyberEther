#include <catch2/catch_test_macros.hpp>

#include "dmi_block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Dynamic Tensor Import block delegates tensor validation "
                 "to its module",
                 "[modules][dynamic_tensor_import][block][validation]") {
    Blocks::DynamicTensorImport config;
    REQUIRE(flowgraph->blockCreate("dti_invalid", config, {}) == Result::SUCCESS);

    const auto block = viewBlock("dti_invalid");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.interfaceOutputs.size() == 1);
    REQUIRE(block.interfaceOutputs.front().name == "buffer");
    REQUIRE(block.outputs.empty());
    REQUIRE(block.diagnostic.find("[MODULE_DYNAMIC_TENSOR_IMPORT]") !=
            std::string::npos);
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Dynamic Tensor Import block preserves an invalid import for recovery",
                 "[modules][dynamic_tensor_import][block][reconfigure][validation]") {
    Tensor buffer;
    REQUIRE(buffer.create(DeviceType::CPU, DataType::F32, {4}) == Result::SUCCESS);
    buffer.at<F32>(1) = 7.0f;

    Blocks::DynamicTensorImport config;
    config.buffer = buffer;
    REQUIRE(flowgraph->blockCreate("dti_update", config, {}) == Result::SUCCESS);

    const auto created = viewBlock("dti_update");
    REQUIRE(created.state == Block::State::Created);
    REQUIRE(created.outputs.at("buffer").tensor.id() == buffer.id());
    REQUIRE(created.outputs.at("buffer").tensor.at<F32>(1) == 7.0f);

    Parser::Map update;
    update["buffer"] = Tensor{};
    REQUIRE(flowgraph->blockReconfigure("dti_update", update) == Result::SUCCESS);

    const auto rejected = viewBlock("dti_update");
    REQUIRE(rejected.state == Block::State::Errored);
    REQUIRE(rejected.outputs.empty());
    REQUIRE(rejected.interfaceOutputs.size() == 1);

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("dti_update", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<Tensor>(saved.at("buffer")).id() == 0);

    Parser::Map recovery;
    recovery["buffer"] = buffer;
    REQUIRE(flowgraph->blockReconfigure("dti_update", recovery) == Result::SUCCESS);
    const auto recovered = viewBlock("dti_update");
    REQUIRE(recovered.state == Block::State::Created);
    REQUIRE(recovered.outputs.at("buffer").tensor.id() == buffer.id());
    REQUIRE(recovered.outputs.at("buffer").tensor.at<F32>(1) == 7.0f);
}
