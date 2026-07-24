#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/core/duplicate/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Duplicate block creates and exposes buffer",
                 "[modules][duplicate][block]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("dup_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("dup_src", "window");

    Blocks::Duplicate config;
    config.hostAccessible = true;
    REQUIRE(flowgraph->blockCreate("dup_block", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("dup_block").state == Block::State::Created);
    REQUIRE(viewBlock("dup_block").outputs.count("buffer") == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "Duplicate block reconnects input",
                 "[modules][duplicate][block][lifecycle]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("dup_life_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("dup_life_src", "window");
    REQUIRE(flowgraph->blockCreate("dup_life", "duplicate", {}, inputs) ==
            Result::SUCCESS);

    REQUIRE(flowgraph->blockDisconnect("dup_life", "buffer") == Result::SUCCESS);
    REQUIRE(viewBlock("dup_life").state == Block::State::Incomplete);

    REQUIRE(flowgraph->blockConnect("dup_life", "buffer", "dup_life_src", "window") ==
            Result::SUCCESS);
    REQUIRE(viewBlock("dup_life").state == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Duplicate block propagates unavailable output backend validation",
                 "[modules][duplicate][block][validation]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("dup_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("dup_bad_src", "window");

    Blocks::Duplicate config;
    config.outputDevice = "webgpu";
    REQUIRE(flowgraph->blockCreate("dup_bad", config, inputs) == Result::SUCCESS);

    const auto block = viewBlock("dup_bad");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.outputs.empty());
    REQUIRE(block.diagnostic.find("[DUPLICATE]") != std::string::npos);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Duplicate block preserves live output after rejected target update",
                 "[modules][duplicate][block][validation][reconfigure]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("dup_update_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("dup_update_src", "window");

    Blocks::Duplicate config;
    REQUIRE(flowgraph->blockCreate("dup_update", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    const auto outputId = viewBlock("dup_update").outputs.at("buffer").tensor.id();

    Parser::Map update;
    update["hostAccessible"] = false;
    update["outputDevice"] = std::string("webgpu");
    REQUIRE(flowgraph->blockReconfigure("dup_update", update) == Result::ERROR);

    const auto block = viewBlock("dup_update");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.at("buffer").tensor.id() == outputId);

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("dup_update", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<std::string>(saved.at("outputDevice")) ==
            config.outputDevice);
    REQUIRE(std::any_cast<bool>(saved.at("hostAccessible")) ==
            config.hostAccessible);

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(viewBlock("dup_update").outputs.at("buffer").tensor.id() == outputId);
}
