#include <catch2/catch_test_macros.hpp>

#include <string>

#include "jetstream/domains/visualization/note/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Note block creates with default and custom content",
                 "[modules][note][block]") {
    REQUIRE(flowgraph->blockCreate("note_default", Blocks::Note(), {}) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("note_default")->state() ==
            Block::State::Created);
    REQUIRE(flowgraph->blockList().at("note_default")->inputs().empty());
    REQUIRE(flowgraph->blockList().at("note_default")->outputs().empty());

    Blocks::Note custom;
    custom.content = "# Title\nA **note** for tests.";
    REQUIRE(flowgraph->blockCreate("note_custom", custom, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("note_custom")->state() ==
            Block::State::Created);

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("note_custom", saved) == Result::SUCCESS);
    REQUIRE(saved.contains("content"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Note block reconfigure keeps created state",
                 "[modules][note][block][reconfigure]") {
    REQUIRE(flowgraph->blockCreate("note", Blocks::Note(), {}) == Result::SUCCESS);

    Parser::Map config;
    config["content"] = std::string("## Updated\nStill markdown.");

    REQUIRE(flowgraph->blockReconfigure("note", config) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("note")->state() == Block::State::Created);
}
