#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/flowgraph.hh"

using namespace Jetstream;

TEST_CASE("Node layout and collapse messages preserve each other in either order",
          "[core][sakura][node_metadata][persistence]") {
    for (const bool collapsed : {false, true}) {
        for (const bool collapseFirst : {false, true}) {
            CAPTURE(collapsed, collapseFirst);
            DefaultCompositorState state;
            DefaultCompositorCallbacks callbacks;
            FlowgraphActions actions(state, callbacks);
            auto flowgraph = std::make_shared<Flowgraph>();
            state.flowgraph.items["graph"] = flowgraph;
            const NodeMeta initial{10.0f, 20.0f, 220.0f, 400.0f, !collapsed};
            REQUIRE(flowgraph->metadata().set("node", initial, "editor") == Result::SUCCESS);
            REQUIRE(flowgraph->metadata().set("node", initial, "other") == Result::SUCCESS);

            const MailSetNodeMeta layout{
                "graph", "editor", NodeMeta{30.0f, 40.0f, 320.0f, 580.0f, !collapsed}};
            const MailSetNodeConfigCollapsed collapse{"graph", "editor", collapsed};
            if (collapseFirst) {
                REQUIRE(actions.handle(collapse) == Result::SUCCESS);
                REQUIRE(actions.handle(layout) == Result::SUCCESS);
            } else {
                REQUIRE(actions.handle(layout) == Result::SUCCESS);
                REQUIRE(actions.handle(collapse) == Result::SUCCESS);
            }

            NodeMeta saved;
            REQUIRE(flowgraph->metadata().get("node", saved, "editor") == Result::SUCCESS);
            REQUIRE(saved.x == 30.0f);
            REQUIRE(saved.y == 40.0f);
            REQUIRE(saved.width == 320.0f);
            REQUIRE(saved.height == 580.0f);
            REQUIRE(saved.configCollapsed == collapsed);

            NodeMeta other;
            REQUIRE(flowgraph->metadata().get("node", other, "other") == Result::SUCCESS);
            REQUIRE(other.x == initial.x);
            REQUIRE(other.y == initial.y);
            REQUIRE(other.width == initial.width);
            REQUIRE(other.height == initial.height);
            REQUIRE(other.configCollapsed == initial.configCollapsed);
        }
    }
}

TEST_CASE("Queued node metadata is harmless after its flowgraph closes",
          "[core][sakura][node_metadata][lifecycle]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    FlowgraphActions actions(state, callbacks);
    REQUIRE(actions.handle(MailSetNodeMeta{"closed", "editor", NodeMeta{}}) == Result::SUCCESS);
    REQUIRE(actions.handle(MailSetNodeConfigCollapsed{"closed", "editor", true}) == Result::SUCCESS);
    REQUIRE(state.flowgraph.items.empty());
}
