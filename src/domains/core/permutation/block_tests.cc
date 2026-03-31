#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/core/permutation/block.hh"
#include "jetstream/domains/core/expand_dims/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Permutation block creates with default 1D configuration",
                 "[modules][permutation][block]") {
    Blocks::Window source;
    source.size = 16;
    REQUIRE(flowgraph->blockCreate("perm_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("perm_src", "window");

    Blocks::Permutation config;
    REQUIRE(flowgraph->blockCreate("perm_block", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("perm_block")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("perm_block")->outputs().count("buffer") == 1);

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("perm_block", saved) == Result::SUCCESS);
    REQUIRE(saved.contains("permutation"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Permutation block creates and reconfigures swapped 2D axes",
                 "[modules][permutation][block][reconfigure]") {
    Blocks::Window source;
    source.size = 16;
    REQUIRE(flowgraph->blockCreate("perm_recfg_src", source, {}) == Result::SUCCESS);

    Blocks::ExpandDims expand;
    expand.axis = 0;

    TensorMap expandInputs;
    expandInputs["buffer"].requested("perm_recfg_src", "window");
    REQUIRE(flowgraph->blockCreate("perm_expand", expand, expandInputs) == Result::SUCCESS);

    Blocks::Permutation config;
    config.permutation = {1, 0};

    TensorMap inputs;
    inputs["buffer"].requested("perm_expand", "buffer");
    REQUIRE(flowgraph->blockCreate("perm_recfg", config, inputs) == Result::SUCCESS);

    const Tensor& initialOut = flowgraph->blockList().at("perm_recfg")->outputs().at("buffer").tensor;
    REQUIRE(initialOut.shape(0) == 16);
    REQUIRE(initialOut.shape(1) == 1);

    Parser::Map update;
    update["permutation"] = std::vector<U64>{0, 1};
    REQUIRE(flowgraph->blockReconfigure("perm_recfg", update) == Result::SUCCESS);

    const Tensor& updatedOut = flowgraph->blockList().at("perm_recfg")->outputs().at("buffer").tensor;
    REQUIRE(updatedOut.shape(0) == 1);
    REQUIRE(updatedOut.shape(1) == 16);
    REQUIRE(flowgraph->blockList().at("perm_recfg")->state() == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Permutation block handles disconnect and reconnect",
                 "[modules][permutation][block][lifecycle]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("perm_life_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("perm_life_src", "window");
    REQUIRE(flowgraph->blockCreate("perm_life", "permutation", {}, inputs) == Result::SUCCESS);

    REQUIRE(flowgraph->blockDisconnect("perm_life", "buffer") == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("perm_life")->state() == Block::State::Incomplete);

    REQUIRE(flowgraph->blockConnect("perm_life", "buffer", "perm_life_src", "window") ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("perm_life")->state() == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Permutation block rejects invalid permutation configuration",
                 "[modules][permutation][block][validation]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("perm_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("perm_bad_src", "window");

    Blocks::Permutation config;
    config.permutation = {1, 1};
    REQUIRE(flowgraph->blockCreate("perm_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("perm_bad")->state() == Block::State::Errored);
}
