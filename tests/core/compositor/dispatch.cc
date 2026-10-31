#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/base.hh"
#include "flowgraph_fixture.hh"

#include <chrono>
#include <deque>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace Jetstream;

namespace {

struct DispatchFixture : FlowgraphFixture {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    AgentRuntime runtime{state, callbacks};
    DefaultActions actions{state, callbacks, runtime};
    const std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("cyberether-dispatch-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".yaml");
    const TileMeta initialLayout{
        .split = "row",
        .children = std::vector<TileMeta>{
            {.weight = 600.0f, .pane = "flowgraph"},
            {.weight = 300.0f, .pane = "description"},
        },
    };
    U64 saves = 0;

    DispatchFixture() {
        state.flowgraph.items["graph"] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});
        state.flowgraph.restoredLayouts.insert("graph");
        REQUIRE(flowgraph->blockCreate("a", TestFlowgraph::kSyntheticFaultType, {}, {}) == Result::SUCCESS);
        REQUIRE(flowgraph->setDescription("Description") == Result::SUCCESS);
        REQUIRE(flowgraph->metadata().set("node", NodeMeta{}, "a") == Result::SUCCESS);
        REQUIRE(flowgraph->metadata().set("editor", EditorMeta{}) == Result::SUCCESS);
        REQUIRE(flowgraph->metadata().set("surface_default", SurfaceMeta{}, "a") == Result::SUCCESS);
        REQUIRE(actions.handle(MailSetFlowgraphLayout{"graph", initialLayout}) == Result::SUCCESS);
        callbacks.enqueueCommandWithMail = [this](Sakura::Callback<Result()> command, bool, Mail) {
            REQUIRE(command() == Result::SUCCESS);
            ++saves;
        };
    }

    ~DispatchFixture() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};

}  // namespace

TEST_CASE_METHOD(DispatchFixture, "Save dispatch captures trailing reports without crossing user actions",
                 "[core][compositor][dispatch][recovery]") {
    auto layout = initialLayout;
    layout.children->back().weight = 520.0f;
    layout.children->back().minimized = true;
    const NodeMeta node{132.5f, -54.25f, 320.0f, 440.0f};
    const EditorMeta editor{-145.5f, 78.25f, 1.75f};
    const SurfaceMeta surface{.attachedHeight = 480};
    std::vector<std::shared_ptr<Module::Surface>> surfaces;
    REQUIRE(flowgraph->view().surfaces("a", surfaces) == Result::SUCCESS);
    REQUIRE(surfaces.size() == 1);
    std::deque<Mail> pending{MailSaveFlowgraph{"graph", Platform::PathToUtf8(path)}};
    bool afterUserAction = false;

    SECTION("reports immediately after save are in the file") {}
    SECTION("reports after another user action stay after the save") {
        pending.emplace_back(MailCloseFlowgraphDescription{"graph"});
        afterUserAction = true;
    }

    pending.emplace_back(MailSetFlowgraphLayout{"graph", layout});
    pending.emplace_back(MailSetNodeMeta{"graph", "a", NodeMeta{.x = 10.0f}});
    pending.emplace_back(MailSetEditorMeta{"graph", editor});
    pending.emplace_back(MailResizeSurface{
        .surface = surfaces.front(),
        .flowgraph = "graph",
        .block = "a",
        .metaKey = "surface_default",
        .resize = {.logicalSize = {640, 480}, .framebufferSize = {1280, 960}, .scale = 2.0f},
    });
    pending.emplace_back(MailSetNodeMeta{"graph", "a", node});
    REQUIRE(actions.dispatch(pending) == Result::SUCCESS);
    REQUIRE(saves == 1);

    Flowgraph loaded;
    REQUIRE(loaded.create({}, nullptr, nullptr, nullptr) == Result::SUCCESS);
    REQUIRE(loaded.importFromFile(Platform::PathToUtf8(path)) == Result::SUCCESS);
    TileMeta storedLayout;
    NodeMeta storedNode;
    EditorMeta storedEditor;
    SurfaceMeta storedSurface;
    REQUIRE(loaded.metadata().get("layout", storedLayout) == Result::SUCCESS);
    REQUIRE(loaded.metadata().get("node", storedNode, "a") == Result::SUCCESS);
    REQUIRE(loaded.metadata().get("editor", storedEditor) == Result::SUCCESS);
    REQUIRE(loaded.metadata().get("surface_default", storedSurface, "a") == Result::SUCCESS);
    CHECK(Parser::Hash(storedLayout) == Parser::Hash(afterUserAction ? initialLayout : layout));
    CHECK(Parser::Hash(storedNode) == Parser::Hash(afterUserAction ? NodeMeta{} : node));
    CHECK(Parser::Hash(storedEditor) == Parser::Hash(afterUserAction ? EditorMeta{} : editor));
    CHECK(Parser::Hash(storedSurface) == Parser::Hash(afterUserAction ? SurfaceMeta{} : surface));
    REQUIRE(loaded.destroy() == Result::SUCCESS);

    REQUIRE(flowgraph->metadata().get("node", storedNode, "a") == Result::SUCCESS);
    CHECK(Parser::Hash(storedNode) == Parser::Hash(node));
    REQUIRE(flowgraph->metadata().get("editor", storedEditor) == Result::SUCCESS);
    CHECK(Parser::Hash(storedEditor) == Parser::Hash(editor));
    REQUIRE(flowgraph->metadata().get("surface_default", storedSurface, "a") == Result::SUCCESS);
    CHECK(Parser::Hash(storedSurface) == Parser::Hash(surface));
    CHECK(state.flowgraph.closedDescriptions.contains("graph") == afterUserAction);
    CHECK(ContainsTilePane(state.flowgraph.layouts.at("graph"), "description") == !afterUserAction);
}
