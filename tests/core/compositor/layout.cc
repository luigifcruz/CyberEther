#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/flowgraph.hh"
#include "compositor/default/actions/layout.hh"
#include "flowgraph_fixture.hh"

#include <memory>
#include <string>

using namespace Jetstream;

namespace {

std::vector<std::string> Panes(const TileMeta& tiles) {
    std::vector<std::string> panes;
    CollectTilePanes(tiles, panes);
    return panes;
}

}  // namespace

TEST_CASE("Layout restoration waits for saved metadata and restores the layout once",
          "[core][compositor][layout][metadata]") {
    DefaultCompositorState::FlowgraphState state;
    LayoutActions actions(state);
    auto flowgraph = std::make_shared<Flowgraph>();
    state.items["graph"] = flowgraph;

    for (int poll = 0; poll < 3; ++poll) {
        actions.restoreFromMetadata();
        CHECK_FALSE(state.restoredLayouts.contains("graph"));
        CHECK_FALSE(state.layouts.contains("graph"));
    }

    const TileMeta layout{
        .split = "row",
        .children = std::vector<TileMeta>{
            {.weight = 640.0f, .pane = "flowgraph"},
            {
                .split = "column",
                .weight = 420.0f,
                .children = std::vector<TileMeta>{
                    {.weight = 1.0f, .pane = "surface:adsb:default"},
                    {.weight = 1.0f, .pane = "config:adsb"},
                },
            },
        },
    };
    REQUIRE(flowgraph->metadata().set("layout", layout) == Result::SUCCESS);

    actions.restoreFromMetadata();
    REQUIRE(state.restoredLayouts.contains("graph"));
    REQUIRE(state.layouts.contains("graph"));
    const auto& restored = state.layouts.at("graph");
    REQUIRE(restored.split == "row");
    REQUIRE(restored.children->size() == 2);
    CHECK(*restored.children->at(0).weight == Catch::Approx(640.0f));
    CHECK(*restored.children->at(1).weight == Catch::Approx(420.0f));
    CHECK(Panes(restored) == std::vector<std::string>{"flowgraph", "surface:adsb:default", "config:adsb"});

    state.layouts["graph"] = TileMeta{.pane = "flowgraph"};
    actions.restoreFromMetadata();
    CHECK(state.layouts.at("graph").pane == "flowgraph");
}

TEST_CASE("Old stack metadata completes restoration, is cleared, and stack panes leave the layout",
          "[core][compositor][layout][metadata]") {
    DefaultCompositorState::FlowgraphState state;
    LayoutActions actions(state);
    auto flowgraph = std::make_shared<Flowgraph>();
    state.items["graph"] = flowgraph;

    Parser::Map savedStacks;
    REQUIRE(Parser::YamlDecode(R"(
stack_0:
  title: Stack 0
  tiles:
    pane: surface:adsb:default
)", savedStacks) == Result::SUCCESS);
    REQUIRE(flowgraph->metadata().set("stacks", savedStacks) == Result::SUCCESS);

    SECTION("without a saved layout") {
        actions.restoreFromMetadata();
        REQUIRE(state.restoredLayouts.contains("graph"));
        CHECK_FALSE(flowgraph->metadata().has("stacks"));
        CHECK_FALSE(state.layouts.contains("graph"));
        CHECK_FALSE(flowgraph->metadata().has("layout"));
    }

    SECTION("with a saved layout that references stacks") {
        const TileMeta layout{
            .split = "row",
            .children = std::vector<TileMeta>{
                {.weight = 0.6f, .pane = "flowgraph"},
                {.weight = 0.4f, .pane = "stack_0"},
            },
        };
        REQUIRE(flowgraph->metadata().set("layout", layout) == Result::SUCCESS);
        actions.restoreFromMetadata();
        CHECK_FALSE(flowgraph->metadata().has("stacks"));
        REQUIRE(state.layouts.contains("graph"));
        CHECK(state.layouts.at("graph").pane == "flowgraph");
        TileMeta saved;
        REQUIRE(flowgraph->metadata().get("layout", saved) == Result::SUCCESS);
        CHECK(saved.pane == "flowgraph");
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Layout restoration places detached panes missing from the saved layout as new columns",
                 "[core][compositor][layout][metadata]") {
    REQUIRE(flowgraph->blockCreate("source", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    DefaultCompositorState::FlowgraphState state;
    LayoutActions actions(state);
    state.items["graph"] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});

    REQUIRE(flowgraph->metadata().set("config", ConfigMeta{.detached = true}, "source") == Result::SUCCESS);
    REQUIRE(flowgraph->metadata().set("surface_default", SurfaceMeta{.detached = true}, "source") == Result::SUCCESS);
    REQUIRE(flowgraph->metadata().set("layout", TileMeta{.pane = "flowgraph"}) == Result::SUCCESS);

    actions.restoreFromMetadata();
    const auto& layout = state.layouts.at("graph");
    REQUIRE(layout.split == "row");
    REQUIRE(layout.children->size() == 3);
    CHECK(layout.children->at(0).pane == "flowgraph");
    const auto panes = Panes(layout);
    CHECK(std::find(panes.begin(), panes.end(), "config:source") != panes.end());
    CHECK(std::find(panes.begin(), panes.end(), "surface:source:default") != panes.end());
    for (const auto& child : *layout.children) {
        CHECK_FALSE(child.weight.has_value());
    }
    TileMeta saved;
    REQUIRE(flowgraph->metadata().get("layout", saved) == Result::SUCCESS);
    CHECK(Parser::Hash(saved) == Parser::Hash(layout));
}

TEST_CASE("Flowgraph layouts stay in memory until restored and are saved afterwards",
          "[core][compositor][layout]") {
    DefaultCompositorState::FlowgraphState state;
    LayoutActions actions(state);
    auto flowgraph = std::make_shared<Flowgraph>();
    state.items["graph"] = flowgraph;

    const TileMeta layout{
        .split = "row",
        .children = std::vector<TileMeta>{
            {.weight = 600.0f, .pane = "flowgraph"},
            {.weight = 400.0f, .pane = "config:source"},
        },
    };

    REQUIRE(actions.handle(MailSetFlowgraphLayout{"graph", layout}) == Result::SUCCESS);
    REQUIRE(Parser::Hash(state.layouts.at("graph")) == Parser::Hash(layout));
    REQUIRE_FALSE(flowgraph->metadata().has("layout"));

    REQUIRE(actions.handle(MailSetFlowgraphLayout{"missing", layout}) == Result::SUCCESS);
    REQUIRE_FALSE(state.layouts.contains("missing"));

    state.restoredLayouts.insert("graph");
    REQUIRE(actions.handle(MailSetFlowgraphLayout{"graph", std::nullopt}) == Result::SUCCESS);
    REQUIRE_FALSE(flowgraph->metadata().has("layout"));

    REQUIRE(actions.handle(MailSetFlowgraphLayout{"graph", layout}) == Result::SUCCESS);
    TileMeta saved;
    REQUIRE(flowgraph->metadata().get("layout", saved) == Result::SUCCESS);
    REQUIRE(saved.children->size() == 2);
    CHECK(*saved.children->at(0).weight == Catch::Approx(600.0f));
    CHECK(saved.children->at(1).pane == "config:source");

    const TileMeta editorOnly{.pane = "flowgraph"};
    REQUIRE(actions.handle(MailSetFlowgraphLayout{"graph", editorOnly}) == Result::SUCCESS);
    TileMeta replaced;
    REQUIRE(flowgraph->metadata().get("layout", replaced) == Result::SUCCESS);
    CHECK(replaced.pane == "flowgraph");
    CHECK_FALSE(replaced.children.has_value());
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Detached items join the layout as new columns and are saved",
                 "[core][compositor][layout][detach]") {
    REQUIRE(flowgraph->blockCreate("source", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    DefaultCompositorState::FlowgraphState state;
    LayoutActions actions(state);
    state.items["graph"] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});

    const auto configKey = MakeConfigTileKey("source");
    const auto surfaceKey = MakeSurfaceTileKey("source", "default");

    REQUIRE(actions.handle(MailSetConfigDetached{"graph", "source", true}) == Result::SUCCESS);
    REQUIRE(state.restoredLayouts.contains("graph"));
    auto layout = state.layouts.at("graph");
    REQUIRE(layout.split == "row");
    REQUIRE(layout.children->size() == 2);
    CHECK(layout.children->at(0).pane == "flowgraph");
    CHECK(layout.children->at(1).pane == configKey);

    REQUIRE(actions.handle(MailSetSurfaceDetached{"graph", "source", "default", true}) == Result::SUCCESS);
    layout = state.layouts.at("graph");
    REQUIRE(layout.children->size() == 3);
    CHECK(layout.children->at(2).pane == surfaceKey);

    REQUIRE(actions.handle(MailSetSurfaceDetached{"graph", "source", "default", true}) == Result::SUCCESS);
    CHECK(state.layouts.at("graph").children->size() == 3);

    TileMeta saved;
    REQUIRE(flowgraph->metadata().get("layout", saved) == Result::SUCCESS);
    CHECK(Parser::Hash(saved) == Parser::Hash(state.layouts.at("graph")));
    CHECK_FALSE(flowgraph->metadata().has("stacks"));

    ConfigMeta config;
    REQUIRE(flowgraph->metadata().get("config", config, "source") == Result::SUCCESS);
    CHECK(config.detached);
    SurfaceMeta surface;
    REQUIRE(flowgraph->metadata().get("surface_default", surface, "source") == Result::SUCCESS);
    CHECK(surface.detached);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Reattached items leave the layout",
                 "[core][compositor][layout][detach]") {
    REQUIRE(flowgraph->blockCreate("source", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    DefaultCompositorState::FlowgraphState state;
    LayoutActions actions(state);
    state.items["graph"] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});

    REQUIRE(actions.handle(MailSetConfigDetached{"graph", "source", true}) == Result::SUCCESS);
    REQUIRE(actions.handle(MailSetSurfaceDetached{"graph", "source", "default", true}) == Result::SUCCESS);

    REQUIRE(actions.handle(MailSetConfigDetached{"graph", "source", false}) == Result::SUCCESS);
    CHECK(Panes(state.layouts.at("graph")) == std::vector<std::string>{"flowgraph", "surface:source:default"});
    ConfigMeta config;
    REQUIRE(flowgraph->metadata().get("config", config, "source") == Result::SUCCESS);
    CHECK_FALSE(config.detached);

    REQUIRE(actions.handle(MailSetSurfaceDetached{"graph", "source", "default", false}) == Result::SUCCESS);
    CHECK(state.layouts.at("graph").pane == "flowgraph");
    TileMeta saved;
    REQUIRE(flowgraph->metadata().get("layout", saved) == Result::SUCCESS);
    CHECK(saved.pane == "flowgraph");
    SurfaceMeta surface;
    REQUIRE(flowgraph->metadata().get("surface_default", surface, "source") == Result::SUCCESS);
    CHECK_FALSE(surface.detached);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Detach requests for unknown blocks change nothing",
                 "[core][compositor][layout][detach]") {
    DefaultCompositorState::FlowgraphState state;
    LayoutActions actions(state);
    state.items["graph"] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});

    REQUIRE(actions.handle(MailSetConfigDetached{"graph", "missing", true}) == Result::SUCCESS);
    REQUIRE(actions.handle(MailSetSurfaceDetached{"graph", "missing", "default", true}) == Result::SUCCESS);
    REQUIRE(actions.handle(MailSetConfigDetached{"other", "missing", true}) == Result::SUCCESS);
    CHECK_FALSE(state.layouts.contains("graph"));
    CHECK_FALSE(state.restoredLayouts.contains("graph"));
    CHECK_FALSE(flowgraph->metadata().has("layout"));
    CHECK_FALSE(flowgraph->metadata().has("config", "missing"));
}

TEST_CASE("Removing a layout block drops its surface and config tiles",
          "[core][compositor][layout][lifecycle]") {
    DefaultCompositorState::FlowgraphState state;
    LayoutActions actions(state);
    auto flowgraph = std::make_shared<Flowgraph>();
    state.items["graph"] = flowgraph;
    state.restoredLayouts.insert("graph");

    state.layouts["graph"] = TileMeta{
        .split = "row",
        .children = std::vector<TileMeta>{
            {.weight = 600.0f, .pane = "flowgraph"},
            {
                .split = "column",
                .weight = 420.0f,
                .children = std::vector<TileMeta>{
                    {.weight = 1.0f, .pane = "surface:source:waterfall"},
                    {.weight = 1.0f, .pane = "config:source"},
                },
            },
            {.weight = 300.0f, .pane = "surface:source2:default"},
        },
    };

    REQUIRE(actions.handle(MailRemoveLayoutBlock{"graph", "source"}) == Result::SUCCESS);
    const auto& layout = state.layouts.at("graph");
    REQUIRE(layout.children->size() == 2);
    CHECK(layout.children->at(0).pane == "flowgraph");
    CHECK(layout.children->at(1).pane == "surface:source2:default");
    CHECK(*layout.children->at(1).weight == Catch::Approx(300.0f));
    TileMeta saved;
    REQUIRE(flowgraph->metadata().get("layout", saved) == Result::SUCCESS);
    CHECK(Parser::Hash(saved) == Parser::Hash(layout));
}

TEST_CASE("Renaming a layout block leaves similarly named blocks alone",
          "[core][compositor][layout][rename]") {
    DefaultCompositorState::FlowgraphState state;
    LayoutActions actions(state);
    auto flowgraph = std::make_shared<Flowgraph>();
    state.items["graph"] = flowgraph;
    state.restoredLayouts.insert("graph");

    state.layouts["graph"] = TileMeta{
        .split = "column",
        .children = std::vector<TileMeta>{
            {.weight = 0.1f, .pane = "config:source"},
            {.weight = 0.2f, .pane = "config:source2"},
            {.weight = 0.3f, .pane = "surface:source:default"},
            {.weight = 0.4f, .pane = "surface:source2:source"},
        },
    };

    REQUIRE(actions.handle(MailRenameLayoutBlock{"graph", "source", "renamed"}) == Result::SUCCESS);
    const auto& children = *state.layouts.at("graph").children;
    REQUIRE(children.size() == 4);
    CHECK(children[0].pane == "config:renamed");
    CHECK(children[1].pane == "config:source2");
    CHECK(children[2].pane == "surface:renamed:default");
    CHECK(children[3].pane == "surface:source2:source");
    CHECK(children[0].weight == 0.1f);
    CHECK(children[3].weight == 0.4f);
    CHECK(flowgraph->metadata().has("layout"));
}

TEST_CASE("Layout edits ignore closed flowgraphs with retained layouts",
          "[core][compositor][layout][lifecycle]") {
    DefaultCompositorState::FlowgraphState state;
    LayoutActions actions(state);
    state.layouts["closed"] = TileMeta{.pane = "config:deleted"};
    const auto before = Parser::Hash(state.layouts.at("closed"));

    REQUIRE(actions.handle(MailRemoveLayoutBlock{"closed", "deleted"}) == Result::SUCCESS);
    CHECK(Parser::Hash(state.layouts.at("closed")) == before);
    REQUIRE(actions.handle(MailRenameLayoutBlock{"closed", "deleted", "renamed"}) == Result::SUCCESS);
    CHECK(Parser::Hash(state.layouts.at("closed")) == before);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Block renames preserve config state and migrate saved layout references",
                 "[core][compositor][layout][rename]") {
    REQUIRE(flowgraph->blockCreate("source", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    state.flowgraph.items["graph"] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});
    LayoutActions layouts(state.flowgraph);
    FlowgraphActions actions(state, callbacks);
    REQUIRE(layouts.handle(MailSetConfigDetached{"graph", "source", true}) == Result::SUCCESS);
    REQUIRE(layouts.handle(MailSetSurfaceDetached{"graph", "source", "default", true}) == Result::SUCCESS);
    const auto layoutHash = Parser::Hash(state.flowgraph.layouts.at("graph"));

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
    REQUIRE(Parser::Hash(state.flowgraph.layouts.at("graph")) == layoutHash);

    const auto result = command();
    if (newName == "invalid.name") {
        REQUIRE(result == Result::ERROR);
        REQUIRE(Parser::Hash(state.flowgraph.layouts.at("graph")) == layoutHash);
        return;
    }
    REQUIRE(result == Result::SUCCESS);
    REQUIRE(layouts.handle(std::get<MailRenameLayoutBlock>(*completion)) == Result::SUCCESS);

    ConfigMeta after;
    REQUIRE(flowgraph->metadata().get("config", after, newName) == Result::SUCCESS);
    CHECK(after.detached);

    TileMeta saved;
    REQUIRE(flowgraph->metadata().get("layout", saved) == Result::SUCCESS);
    CHECK(Panes(saved) == std::vector<std::string>{"flowgraph", "config:" + newName, "surface:" + newName + ":default"});
}

TEST_CASE("Layout restoration erases a saved layout whose panes are all unknown",
          "[core][compositor][layout][metadata]") {
    DefaultCompositorState::FlowgraphState state;
    LayoutActions actions(state);
    auto flowgraph = std::make_shared<Flowgraph>();
    state.items["graph"] = flowgraph;
    state.layouts["graph"] = TileMeta{.pane = "flowgraph"};

    const TileMeta layout{
        .split = "row",
        .children = std::vector<TileMeta>{
            {.weight = 0.5f, .pane = "stack_0"},
            {.weight = 0.5f, .pane = "inspector"},
        },
    };
    REQUIRE(flowgraph->metadata().set("layout", layout) == Result::SUCCESS);

    actions.restoreFromMetadata();
    CHECK(state.restoredLayouts.contains("graph"));
    CHECK_FALSE(state.layouts.contains("graph"));
    CHECK_FALSE(flowgraph->metadata().has("layout"));
}

TEST_CASE("Layout restoration replaces unreadable layout metadata with the layout in memory",
          "[core][compositor][layout][metadata]") {
    DefaultCompositorState::FlowgraphState state;
    LayoutActions actions(state);
    auto flowgraph = std::make_shared<Flowgraph>();
    state.items["graph"] = flowgraph;

    Parser::Map corrupt;
    REQUIRE(Parser::YamlDecode(R"(
split: row
children: none
)", corrupt) == Result::SUCCESS);
    REQUIRE(flowgraph->metadata().set("layout", corrupt) == Result::SUCCESS);
    TileMeta unreadable;
    REQUIRE(flowgraph->metadata().get("layout", unreadable) != Result::SUCCESS);

    SECTION("without a layout in memory") {
        actions.restoreFromMetadata();
        CHECK(state.restoredLayouts.contains("graph"));
        CHECK_FALSE(state.layouts.contains("graph"));
        CHECK_FALSE(flowgraph->metadata().has("layout"));
    }

    SECTION("with a layout in memory") {
        const TileMeta live{
            .split = "row",
            .children = std::vector<TileMeta>{
                {.weight = 0.7f, .pane = "flowgraph"},
                {.weight = 0.3f, .pane = "config:source"},
            },
        };
        REQUIRE(actions.handle(MailSetFlowgraphLayout{"graph", live}) == Result::SUCCESS);
        actions.restoreFromMetadata();
        CHECK(state.restoredLayouts.contains("graph"));
        REQUIRE(state.layouts.contains("graph"));
        CHECK(Parser::Hash(state.layouts.at("graph")) == Parser::Hash(live));
        TileMeta saved;
        REQUIRE(flowgraph->metadata().get("layout", saved) == Result::SUCCESS);
        CHECK(Parser::Hash(saved) == Parser::Hash(live));
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Layout restoration skips flowgraphs whose detached items have no saved layout",
                 "[core][compositor][layout][metadata]") {
    REQUIRE(flowgraph->blockCreate("source", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    DefaultCompositorState::FlowgraphState state;
    LayoutActions actions(state);
    state.items["graph"] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});

    REQUIRE(flowgraph->metadata().set("config", ConfigMeta{.detached = true}, "source") == Result::SUCCESS);
    REQUIRE(flowgraph->metadata().set("surface_default", SurfaceMeta{.detached = true}, "source") == Result::SUCCESS);

    for (int poll = 0; poll < 3; ++poll) {
        actions.restoreFromMetadata();
        CHECK_FALSE(state.restoredLayouts.contains("graph"));
        CHECK_FALSE(state.layouts.contains("graph"));
        CHECK_FALSE(flowgraph->metadata().has("layout"));
    }

    ConfigMeta config;
    REQUIRE(flowgraph->metadata().get("config", config, "source") == Result::SUCCESS);
    CHECK(config.detached);
    SurfaceMeta surface;
    REQUIRE(flowgraph->metadata().get("surface_default", surface, "source") == Result::SUCCESS);
    CHECK(surface.detached);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Surface config collapse is saved per surface and only when it changes",
                 "[core][compositor][layout][collapse]") {
    REQUIRE(flowgraph->blockCreate("source", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    DefaultCompositorState::FlowgraphState state;
    LayoutActions actions(state);
    state.items["graph"] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});

    REQUIRE(actions.handle(MailSetSurfaceConfigCollapsed{"graph", "source", "default", true}) == Result::SUCCESS);
    CHECK_FALSE(flowgraph->metadata().has("surface_default", "source"));

    REQUIRE(actions.handle(MailSetSurfaceConfigCollapsed{"graph", "source", "default", false}) == Result::SUCCESS);
    SurfaceMeta surface;
    REQUIRE(flowgraph->metadata().get("surface_default", surface, "source") == Result::SUCCESS);
    CHECK_FALSE(surface.configCollapsed);
    CHECK_FALSE(surface.detached);
    CHECK(surface.attachedHeight == 256);
    CHECK_FALSE(flowgraph->metadata().has("surface_waterfall", "source"));

    REQUIRE(flowgraph->metadata().set("surface_waterfall",
                                      SurfaceMeta{.attachedHeight = 320, .detached = true, .configCollapsed = false},
                                      "source") == Result::SUCCESS);
    Parser::Map before;
    REQUIRE(flowgraph->metadata().get("surface_waterfall", before, "source") == Result::SUCCESS);
    REQUIRE(actions.handle(MailSetSurfaceConfigCollapsed{"graph", "source", "waterfall", false}) == Result::SUCCESS);
    Parser::Map unchanged;
    REQUIRE(flowgraph->metadata().get("surface_waterfall", unchanged, "source") == Result::SUCCESS);
    CHECK(Parser::Hash(unchanged) == Parser::Hash(before));

    REQUIRE(actions.handle(MailSetSurfaceConfigCollapsed{"graph", "source", "waterfall", true}) == Result::SUCCESS);
    SurfaceMeta waterfall;
    REQUIRE(flowgraph->metadata().get("surface_waterfall", waterfall, "source") == Result::SUCCESS);
    CHECK(waterfall.configCollapsed);
    CHECK(waterfall.detached);
    CHECK(waterfall.attachedHeight == 320);
    REQUIRE(flowgraph->metadata().get("surface_default", surface, "source") == Result::SUCCESS);
    CHECK_FALSE(surface.configCollapsed);
    CHECK_FALSE(state.layouts.contains("graph"));
    CHECK_FALSE(state.restoredLayouts.contains("graph"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Surface config collapse requests for unknown blocks change nothing",
                 "[core][compositor][layout][collapse]") {
    DefaultCompositorState::FlowgraphState state;
    LayoutActions actions(state);
    state.items["graph"] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});

    REQUIRE(actions.handle(MailSetSurfaceConfigCollapsed{"graph", "missing", "default", false}) == Result::SUCCESS);
    REQUIRE(actions.handle(MailSetSurfaceConfigCollapsed{"other", "missing", "default", false}) == Result::SUCCESS);
    CHECK_FALSE(flowgraph->metadata().has("surface_default", "missing"));
    CHECK_FALSE(state.layouts.contains("graph"));
    CHECK_FALSE(state.restoredLayouts.contains("graph"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Documentation reads the block from the flowgraph and joins the layout as one unsaved column",
                 "[core][compositor][layout][documentation]") {
    REQUIRE(flowgraph->blockCreate("fft", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("filter", TestFlowgraph::kSyntheticIsolatedType, {}, {}) == Result::SUCCESS);
    DefaultCompositorState::FlowgraphState state;
    LayoutActions actions(state);
    state.items["graph"] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});
    REQUIRE(state.blocks.empty());

    REQUIRE(actions.handle(MailOpenDocumentation{"graph", "missing"}) == Result::SUCCESS);
    CHECK_FALSE(state.documentation.contains("graph"));
    CHECK_FALSE(state.layouts.contains("graph"));

    REQUIRE(actions.handle(MailOpenDocumentation{"graph", "fft"}) == Result::SUCCESS);
    REQUIRE(state.documentation.contains("graph"));
    CHECK(state.documentation.at("graph").block == "fft");
    CHECK(state.documentation.at("graph").title == "Synthetic Source");
    CHECK(state.documentation.at("graph").markdown == viewBlock("fft").description);
    CHECK_FALSE(state.documentation.at("graph").markdown.empty());
    CHECK(Panes(state.layouts.at("graph")) == std::vector<std::string>{"flowgraph", "documentation"});
    TileMeta saved;
    REQUIRE(flowgraph->metadata().get("layout", saved) == Result::SUCCESS);
    CHECK(Panes(saved) == std::vector<std::string>{"flowgraph"});

    REQUIRE(actions.handle(MailOpenDocumentation{"graph", "filter"}) == Result::SUCCESS);
    CHECK(state.documentation.at("graph").block == "filter");
    CHECK(state.documentation.at("graph").title == "Synthetic Isolated");
    CHECK(state.documentation.at("graph").markdown == viewBlock("filter").description);
    CHECK(Panes(state.layouts.at("graph")) == std::vector<std::string>{"flowgraph", "documentation"});

    const TileMeta moved{
        .split = "row",
        .children = std::vector<TileMeta>{
            {.weight = 300.0f, .pane = "documentation"},
            {.weight = 700.0f, .pane = "flowgraph"},
        },
    };
    REQUIRE(actions.handle(MailSetFlowgraphLayout{.flowgraph = "graph", .tiles = moved}) == Result::SUCCESS);
    CHECK(Panes(state.layouts.at("graph")) == std::vector<std::string>{"documentation", "flowgraph"});
    REQUIRE(flowgraph->metadata().get("layout", saved) == Result::SUCCESS);
    CHECK(Panes(saved) == std::vector<std::string>{"flowgraph"});

    REQUIRE(actions.handle(MailCloseDocumentation{"graph"}) == Result::SUCCESS);
    CHECK_FALSE(state.documentation.contains("graph"));
    CHECK(Panes(state.layouts.at("graph")) == std::vector<std::string>{"flowgraph"});

    REQUIRE(actions.handle(MailSetFlowgraphLayout{.flowgraph = "graph", .tiles = moved}) == Result::SUCCESS);
    CHECK(Panes(state.layouts.at("graph")) == std::vector<std::string>{"flowgraph"});
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "A file description tile is included in the saved layout",
                 "[core][compositor][layout][description]") {
    DefaultCompositorState::FlowgraphState state;
    LayoutActions actions(state);
    state.items["graph"] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});
    state.restoredLayouts.insert("graph");
    const TileMeta tiles{
        .split = "row",
        .children = std::vector<TileMeta>{
            {.weight = 300.0f, .pane = "description"},
            {.weight = 700.0f, .pane = "flowgraph"},
        },
    };

    REQUIRE(actions.handle(MailSetFlowgraphLayout{.flowgraph = "graph", .tiles = tiles}) == Result::SUCCESS);
    CHECK(Panes(state.layouts.at("graph")) == std::vector<std::string>{"flowgraph"});

    REQUIRE(flowgraph->setDescription("## Overview") == Result::SUCCESS);
    REQUIRE(actions.handle(MailSetFlowgraphLayout{.flowgraph = "graph", .tiles = tiles}) == Result::SUCCESS);
    CHECK(Panes(state.layouts.at("graph")) == std::vector<std::string>{"description", "flowgraph"});
    TileMeta saved;
    REQUIRE(flowgraph->metadata().get("layout", saved) == Result::SUCCESS);
    CHECK(Panes(saved) == std::vector<std::string>{"description", "flowgraph"});
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Closing the file description drops it and its tile from the session layout",
                 "[core][compositor][layout][description]") {
    DefaultCompositorState::FlowgraphState state;
    LayoutActions actions(state);
    state.items["graph"] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});
    state.restoredLayouts.insert("graph");
    REQUIRE(flowgraph->setDescription("## Overview") == Result::SUCCESS);
    const TileMeta tiles{
        .split = "row",
        .children = std::vector<TileMeta>{
            {.weight = 300.0f, .pane = "description"},
            {.weight = 700.0f, .pane = "flowgraph"},
        },
    };
    REQUIRE(actions.handle(MailSetFlowgraphLayout{.flowgraph = "graph", .tiles = tiles}) == Result::SUCCESS);
    REQUIRE(Panes(state.layouts.at("graph")) == std::vector<std::string>{"description", "flowgraph"});

    REQUIRE(actions.handle(MailCloseFlowgraphDescription{.flowgraph = "graph"}) == Result::SUCCESS);
    CHECK(state.closedDescriptions.contains("graph"));
    CHECK(flowgraph->description() == "## Overview");
    CHECK(Panes(state.layouts.at("graph")) == std::vector<std::string>{"flowgraph"});

    REQUIRE(actions.handle(MailSetFlowgraphLayout{.flowgraph = "graph", .tiles = tiles}) == Result::SUCCESS);
    CHECK(Panes(state.layouts.at("graph")) == std::vector<std::string>{"flowgraph"});
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Documentation follows a renamed block and closes when that block is deleted",
                 "[core][compositor][layout][documentation]") {
    REQUIRE(flowgraph->blockCreate("fft", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("filter", TestFlowgraph::kSyntheticIsolatedType, {}, {}) == Result::SUCCESS);
    DefaultCompositorState::FlowgraphState state;
    LayoutActions actions(state);
    state.items["graph"] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});
    state.restoredLayouts.insert("graph");
    REQUIRE(actions.handle(MailOpenDocumentation{"graph", "fft"}) == Result::SUCCESS);

    REQUIRE(actions.handle(MailRenameLayoutBlock{"graph", "fft", "spectrum"}) == Result::SUCCESS);
    CHECK(state.documentation.at("graph").block == "spectrum");
    CHECK(Panes(state.layouts.at("graph")) == std::vector<std::string>{"flowgraph", "documentation"});

    REQUIRE(actions.handle(MailRemoveLayoutBlock{"graph", "filter"}) == Result::SUCCESS);
    CHECK(state.documentation.contains("graph"));
    CHECK(Panes(state.layouts.at("graph")) == std::vector<std::string>{"flowgraph", "documentation"});

    REQUIRE(actions.handle(MailRemoveLayoutBlock{"graph", "spectrum"}) == Result::SUCCESS);
    CHECK_FALSE(state.documentation.contains("graph"));
    CHECK(Panes(state.layouts.at("graph")) == std::vector<std::string>{"flowgraph"});
}

TEST_CASE("Layout restoration drops a documentation tile found in saved metadata",
          "[core][compositor][layout][documentation]") {
    DefaultCompositorState::FlowgraphState state;
    LayoutActions actions(state);
    auto flowgraph = std::make_shared<Flowgraph>();
    state.items["graph"] = flowgraph;
    const TileMeta layout{
        .split = "row",
        .children = std::vector<TileMeta>{
            {.pane = "flowgraph"},
            {.pane = "documentation"},
        },
    };
    REQUIRE(flowgraph->metadata().set("layout", layout) == Result::SUCCESS);

    actions.restoreFromMetadata();
    CHECK(Panes(state.layouts.at("graph")) == std::vector<std::string>{"flowgraph"});
}
