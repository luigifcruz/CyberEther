#include <catch2/catch_test_macros.hpp>

#include "compositor/default/presenters/flowgraph/environment.hh"
#include "compositor/default/presenters/flowgraph/metadata.hh"
#include "flowgraph_fixture.hh"
#include "harness.hh"

#include <imgui_internal.h>

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Key-value trees keep metadata identities and labels distinct",
                 "[core][sakura][compositor][key-value]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    // The fixture owns the graph and handles its lifecycle after state is destroyed.
    state.flowgraph.items["graph"] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});
    state.interface.focusedFlowgraph = "graph";
    state.interface.flowgraphMetadataVisible = true;
    state.interface.flowgraphEnvironmentVisible = true;
    const PresenterContext context{state, callbacks};

    const Parser::Map firstValue{{"child-A", "first value"}};
    const Parser::Map secondValue{{"child-B", "second value"}};
    bool environment = false;
    std::string firstLabel;
    std::string secondLabel;

    SECTION("slashes in block names and keys do not alias") {
        REQUIRE(flowgraph->blockCreate("a", TestFlowgraph::kSyntheticIsolatedType, {}, {}) == Result::SUCCESS);
        REQUIRE(flowgraph->blockCreate("a/b", TestFlowgraph::kSyntheticIsolatedType, {}, {}) == Result::SUCCESS);
        REQUIRE(flowgraph->metadata().set("b/c", firstValue, "a") == Result::SUCCESS);
        REQUIRE(flowgraph->metadata().set("c", secondValue, "a/b") == Result::SUCCESS);
        firstLabel = "a.b/c";
        secondLabel = "a/b.c";
    }

    SECTION("flowgraph and block metadata may have identical display labels") {
        REQUIRE(flowgraph->blockCreate("a", TestFlowgraph::kSyntheticIsolatedType, {}, {}) == Result::SUCCESS);
        REQUIRE(flowgraph->metadata().set("a.metadata", firstValue) == Result::SUCCESS);
        REQUIRE(flowgraph->metadata().set("metadata", secondValue, "a") == Result::SUCCESS);
        firstLabel = secondLabel = "a.metadata";
    }

    SECTION("metadata labels containing ImGui markers remain literal") {
        firstLabel = "alpha##literal###suffix";
        secondLabel = "beta##literal###suffix";
        REQUIRE(flowgraph->metadata().set(firstLabel, firstValue) == Result::SUCCESS);
        REQUIRE(flowgraph->metadata().set(secondLabel, secondValue) == Result::SUCCESS);
    }

    SECTION("environment labels containing ImGui markers remain literal") {
        environment = true;
        firstLabel = "alpha##literal###suffix";
        secondLabel = "beta##literal###suffix";
        REQUIRE(flowgraph->environment().set(firstLabel, firstValue) == Result::SUCCESS);
        REQUIRE(flowgraph->environment().set(secondLabel, secondValue) == Result::SUCCESS);
    }

    const auto build = [&] {
        return environment ? FlowgraphEnvironmentWindowPresenter(context).build() :
                             FlowgraphMetadataWindowPresenter(context).build();
    };
    const auto initial = build();
    REQUIRE(initial.has_value());
    REQUIRE(initial->nodes.size() == 2);
    REQUIRE(initial->nodes[0].id != initial->nodes[1].id);
    REQUIRE(initial->nodes[0].label == firstLabel);
    REQUIRE(initial->nodes[1].label == secondLabel);
    const std::string firstChild = initial->nodes[0].children.at(0).label;
    const std::string secondChild = initial->nodes[1].children.at(0).label;

    SakuraTest::HeadlessUi ui;
    Sakura::Table table;
    table.update({
        .id = "key-value-tree",
        .columns = {"Key", "Value"},
        .fixedColumnWidths = {180.0f},
        .showHeaders = false,
    });
    const auto frame = [&] {
        const auto config = build();
        REQUIRE(config.has_value());
        std::string text;
        ui.frame([&] {
            ImGui::LogToBuffer();
            // Logging must observe, rather than automatically expand, the tree.
            ImGui::GetCurrentContext()->LogDepthToExpand = 0;
            table.render(ui.sakura(), config->nodes);
            text = ImGui::GetCurrentContext()->LogBuffer.c_str();
            ImGui::LogFinish();
        });
        return text;
    };
    frame();
    const auto before = frame();
    REQUIRE(before.find(firstLabel) != std::string::npos);
    REQUIRE(before.find(secondLabel) != std::string::npos);
    REQUIRE(before.find(firstChild) != std::string::npos);
    REQUIRE(before.find(secondChild) != std::string::npos);

    ui.setMouse({10.0f, 8.0f}, false);
    frame();
    ui.setMouse({10.0f, 8.0f}, true);
    frame();
    ui.setMouse({10.0f, 8.0f}, false);
    const auto after = frame();
    REQUIRE(after.find(firstChild) == std::string::npos);
    REQUIRE(after.find(secondChild) != std::string::npos);

    // Rebuilding and filtering the presenter must preserve the other entry's state.
    if (environment) {
        state.interface.flowgraphEnvironmentSearch = secondLabel;
    } else {
        state.interface.flowgraphMetadataSearch = secondLabel;
    }
    REQUIRE(frame().find(secondChild) != std::string::npos);
}
