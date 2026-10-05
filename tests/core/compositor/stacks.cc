#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/stacks.hh"
#include "compositor/default/actions/flowgraph.hh"
#include "compositor/default/presenters/flowgraph/labels.hh"
#include "flowgraph_fixture.hh"

#include <memory>
#include <string>

using namespace Jetstream;

TEST_CASE("Stack restoration waits for saved metadata and restores the layout once",
          "[core][compositor][stacks][metadata]") {
    DefaultCompositorState::FlowgraphState state;
    DefaultCompositorCallbacks callbacks;
    StackActions actions(state, callbacks);
    auto flowgraph = std::make_shared<Flowgraph>();
    state.items["graph"] = flowgraph;

    SECTION("metadata is available on the first poll") {}

    SECTION("metadata arrives after early polls during import") {
        for (int poll = 0; poll < 3; ++poll) {
            actions.restoreFromMetadata();
            CHECK_FALSE(state.stacks.contains("graph"));
        }
    }

    // Saved layout from test23.yml, independent of its SDR-backed blocks.
    Parser::Map savedStacks;
    REQUIRE(Parser::YamlDecode(R"(
stack_0:
  title: Stack 0
  x: 0
  y: 30
  width: 1920
  height: 1050
  layout:
    direction: left
    ratio: 0.75208336
    children:
      - surfaces:
          - block: adsb
            surface: default
            order: 0
      - surfaces:
          - block: spectrum_analyzer
            surface: default
            order: 0
)", savedStacks) == Result::SUCCESS);
    REQUIRE(flowgraph->metadata().set("stacks", savedStacks) == Result::SUCCESS);

    actions.restoreFromMetadata();
    REQUIRE(state.stacks.contains("graph"));
    auto& stacks = state.stacks.at("graph");
    REQUIRE(stacks.size() == 1);
    REQUIRE(stacks.contains("stack_0"));
    auto& stack = stacks.at("stack_0");
    REQUIRE(stack.meta.title == "Stack 0");
    REQUIRE(stack.meta.x == 0.0f);
    REQUIRE(stack.meta.y == 30.0f);
    REQUIRE(stack.meta.width == 1920.0f);
    REQUIRE(stack.meta.height == 1050.0f);
    REQUIRE(stack.restoreDockLayout);
    REQUIRE(stack.dockInMainDockspace);

    REQUIRE(stack.meta.layout.has_value());
    const auto& layout = *stack.meta.layout;
    REQUIRE(layout.direction == "left");
    REQUIRE(layout.ratio.has_value());
    REQUIRE(*layout.ratio == Catch::Approx(0.75208336f));
    REQUIRE(layout.children.has_value());
    REQUIRE(layout.children->size() == 2);
    const auto& left = layout.children->at(0);
    const auto& right = layout.children->at(1);
    REQUIRE(left.surfaces.has_value());
    REQUIRE(right.surfaces.has_value());
    REQUIRE(left.surfaces->size() == 1);
    REQUIRE(right.surfaces->size() == 1);
    REQUIRE(left.surfaces->at(0).block == "adsb");
    REQUIRE(right.surfaces->at(0).block == "spectrum_analyzer");
    REQUIRE(left.surfaces->at(0).surface == "default");
    REQUIRE(right.surfaces->at(0).surface == "default");
    REQUIRE(left.surfaces->at(0).order == 0);
    REQUIRE(right.surfaces->at(0).order == 0);

    // Once the UI consumes restoration, polling must preserve its live state.
    stack.restoreDockLayout = false;
    stack.meta.title = "Edited stack";
    actions.restoreFromMetadata();
    REQUIRE(state.stacks.at("graph").size() == 1);
    REQUIRE_FALSE(state.stacks.at("graph").at("stack_0").restoreDockLayout);
    REQUIRE(state.stacks.at("graph").at("stack_0").meta.title == "Edited stack");
}

TEST_CASE("An explicitly empty stack map completes restoration",
          "[core][compositor][stacks][metadata]") {
    DefaultCompositorState::FlowgraphState state;
    DefaultCompositorCallbacks callbacks;
    StackActions actions(state, callbacks);
    auto flowgraph = std::make_shared<Flowgraph>();
    state.items["graph"] = flowgraph;
    REQUIRE(flowgraph->metadata().set("stacks", Parser::Map{}) == Result::SUCCESS);

    actions.restoreFromMetadata();
    REQUIRE(state.stacks.contains("graph"));
    REQUIRE(state.stacks.at("graph").empty());

    actions.restoreFromMetadata();
    REQUIRE(state.stacks.at("graph").empty());
}

TEST_CASE("Stack restoration skips malformed entries and supplies missing titles",
          "[core][compositor][stacks][metadata]") {
    DefaultCompositorState::FlowgraphState state;
    DefaultCompositorCallbacks callbacks;
    StackActions actions(state, callbacks);
    auto flowgraph = std::make_shared<Flowgraph>();
    state.items["graph"] = flowgraph;

    Parser::Map savedStacks;
    savedStacks[""] = Parser::Map{};
    savedStacks["not_a_map"] = std::string("invalid");
    Parser::Map invalidStack;
    invalidStack["width"] = std::string("invalid");
    savedStacks["invalid_width"] = invalidStack;
    savedStacks["stack_0"] = Parser::Map{};
    REQUIRE(flowgraph->metadata().set("stacks", savedStacks) == Result::SUCCESS);

    actions.restoreFromMetadata();
    REQUIRE(state.stacks.contains("graph"));
    REQUIRE(state.stacks.at("graph").size() == 1);
    REQUIRE(state.stacks.at("graph").contains("stack_0"));
    const auto& stack = state.stacks.at("graph").at("stack_0");
    REQUIRE(stack.meta.title == "stack_0");
    REQUIRE_FALSE(stack.restoreDockLayout);
    REQUIRE(stack.dockInMainDockspace);
}

TEST_CASE("Delayed stack cleanup ignores closed flowgraphs with retained layouts",
          "[core][compositor][stacks][lifecycle]") {
    DefaultCompositorState::FlowgraphState state;
    DefaultCompositorCallbacks callbacks;
    StackActions actions(state, callbacks);
    auto& stack = state.stacks["closed"]["stack_0"];
    stack.meta.layout = StackDockLayoutMeta{
        .configs = std::vector<StackDockConfigMeta>{{.block = "deleted"}},
    };
    const auto before = Parser::Hash(stack.meta);

    REQUIRE(actions.handle(MailRemoveStackBlock{"closed", "deleted"}) == Result::SUCCESS);
    REQUIRE(Parser::Hash(stack.meta) == before);
    REQUIRE(actions.handle(MailRenameStackBlock{"closed", "deleted", "renamed"}) == Result::SUCCESS);
    REQUIRE(Parser::Hash(stack.meta) == before);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Block renames preserve config identity and migrate saved stack references",
                 "[core][compositor][stacks][rename]") {
    REQUIRE(flowgraph->blockCreate("source", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    state.flowgraph.items["graph"] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});
    StackActions stacks(state.flowgraph, callbacks);
    FlowgraphActions actions(state, callbacks);
    auto& stack = state.flowgraph.stacks["graph"]["stack_0"];
    stack.meta.layout = StackDockLayoutMeta{
        .direction = "left",
        .ratio = 0.3f,
        .children = std::vector<StackDockLayoutMeta>{
            {.configs = std::vector<StackDockConfigMeta>{{"other", 0}, {"source", 2}}},
            {.surfaces = std::vector<StackDockSurfaceMeta>{{"source", "default", 1}}},
        },
    };
    REQUIRE(stacks.persistFlowgraphStacks("graph") == Result::SUCCESS);
    REQUIRE(stacks.handle(MailSetConfigDetached{"graph", "source", true}) == Result::SUCCESS);

    ConfigMeta before;
    REQUIRE(flowgraph->metadata().get("config", before, "source") == Result::SUCCESS);
    REQUIRE(before.detached);
    REQUIRE_FALSE(before.windowId.empty());
    const auto windowId = MakeDetachedConfigWindowId("graph", "source", before);
    const auto layoutHash = Parser::Hash(stack.meta.layout);

    std::function<Result()> command;
    std::optional<Mail> completion;
    callbacks.enqueueMail = [](Mail&&) {};
    callbacks.enqueueCommandWithMail = [&](std::function<Result()> fn, bool, Mail mail) {
        command = std::move(fn);
        completion = std::move(mail);
    };

    std::string newName = "renamed";
    SECTION("successful rename") {}
    SECTION("failed rename") { newName = "invalid.name"; }
    SECTION("unchanged name") { newName = "source"; }
    REQUIRE(actions.handle(MailRenameBlock{"graph", "source", newName}) == Result::SUCCESS);
    REQUIRE(command);
    REQUIRE(completion.has_value());
    REQUIRE(Parser::Hash(stack.meta.layout) == layoutHash);

    const auto result = command();
    if (newName == "invalid.name") {
        REQUIRE(result == Result::ERROR);
        REQUIRE(Parser::Hash(stack.meta.layout) == layoutHash);
        REQUIRE_FALSE(stack.restoreDockLayout);
        ConfigMeta after;
        REQUIRE(flowgraph->metadata().get("config", after, "source") == Result::SUCCESS);
        REQUIRE(MakeDetachedConfigWindowId("graph", "source", after) == windowId);
        return;
    }
    REQUIRE(result == Result::SUCCESS);
    REQUIRE(stacks.handle(std::get<MailRenameStackBlock>(*completion)) == Result::SUCCESS);

    // Geometry updates must not overwrite the identity copied during rename.
    REQUIRE(actions.handle(MailSetNodeMeta{"graph", newName, NodeMeta{1.0f, 2.0f}}) == Result::SUCCESS);
    ConfigMeta after;
    REQUIRE(flowgraph->metadata().get("config", after, newName) == Result::SUCCESS);
    REQUIRE(MakeDetachedConfigWindowId("graph", newName, after) == windowId);
    REQUIRE(stack.restoreDockLayout == (newName != "source"));

    Parser::Map saved;
    REQUIRE(flowgraph->metadata().get("stacks", saved) == Result::SUCCESS);
    StackMeta restored;
    REQUIRE(restored.deserialize(std::any_cast<const Parser::Map&>(saved.at("stack_0"))) == Result::SUCCESS);
    REQUIRE(restored.layout->direction == "left");
    REQUIRE(restored.layout->ratio == Catch::Approx(0.3f));
    const auto& children = *restored.layout->children;
    REQUIRE(children.size() == 2);
    REQUIRE(children[0].configs->at(0).block == "other");
    REQUIRE(children[0].configs->at(1).block == newName);
    REQUIRE(children[0].configs->at(1).order == 2);
    REQUIRE(children[1].surfaces->at(0).block == newName);
    REQUIRE(children[1].surfaces->at(0).surface == "default");
    REQUIRE(children[1].surfaces->at(0).order == 1);

    if (newName != "source") {
        REQUIRE(flowgraph->blockCreate("source", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
        REQUIRE(stacks.handle(MailSetConfigDetached{"graph", "source", true}) == Result::SUCCESS);
        ConfigMeta reused;
        REQUIRE(flowgraph->metadata().get("config", reused, "source") == Result::SUCCESS);
        REQUIRE(MakeDetachedConfigWindowId("graph", "source", reused) != windowId);
    }
}
