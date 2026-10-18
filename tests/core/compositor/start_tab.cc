#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/workbench.hh"

#include <memory>

using namespace Jetstream;

TEST_CASE("Start tab opens once per request and closes when a flowgraph is added",
          "[core][compositor][start-tab]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    WorkbenchActions actions(state, callbacks);
    state.flowgraph.items["a"] = std::make_shared<Flowgraph>();

    actions.handle(MailOpenStartTab{});
    CHECK(state.interface.startTab);
    CHECK(state.interface.startTabRequests == 1);
    actions.handle(MailOpenStartTab{});
    CHECK(state.interface.startTabRequests == 2);
    actions.settleStartTab();
    CHECK(state.interface.startTab);

    state.flowgraph.items.erase("a");
    actions.settleStartTab();
    CHECK(state.interface.startTab);
    state.flowgraph.items["b"] = std::make_shared<Flowgraph>();
    actions.settleStartTab();
    CHECK_FALSE(state.interface.startTab);

    actions.handle(MailOpenStartTab{});
    CHECK(state.interface.startTab);
    actions.handle(MailCloseStartTab{});
    CHECK_FALSE(state.interface.startTab);
    state.flowgraph.items["c"] = std::make_shared<Flowgraph>();
    actions.settleStartTab();
    CHECK_FALSE(state.interface.startTab);
}

TEST_CASE("Repeated start tab requests keep the flowgraph count of the first request",
          "[core][compositor][start-tab]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    WorkbenchActions actions(state, callbacks);
    state.flowgraph.items["a"] = std::make_shared<Flowgraph>();

    actions.handle(MailOpenStartTab{});
    CHECK(state.interface.startTabBaseline == 1);
    state.flowgraph.items["b"] = std::make_shared<Flowgraph>();
    actions.handle(MailOpenStartTab{});
    CHECK(state.interface.startTabRequests == 2);
    CHECK(state.interface.startTabBaseline == 1);
    actions.settleStartTab();
    CHECK_FALSE(state.interface.startTab);
}

TEST_CASE("Start tab tracks whether it is the focused workspace pane",
          "[core][compositor][start-tab]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    WorkbenchActions actions(state, callbacks);
    state.flowgraph.items["a"] = std::make_shared<Flowgraph>();

    actions.handle(MailFocusStartTab{});
    CHECK_FALSE(state.interface.startTabFocused);

    actions.handle(MailOpenStartTab{});
    CHECK(state.interface.startTabFocused);

    state.interface.startTabFocused = false;
    actions.handle(MailFocusStartTab{});
    CHECK(state.interface.startTabFocused);

    actions.handle(MailCloseStartTab{});
    CHECK_FALSE(state.interface.startTabFocused);

    actions.handle(MailOpenStartTab{});
    state.flowgraph.items["b"] = std::make_shared<Flowgraph>();
    actions.settleStartTab();
    CHECK_FALSE(state.interface.startTab);
    CHECK_FALSE(state.interface.startTabFocused);
}
