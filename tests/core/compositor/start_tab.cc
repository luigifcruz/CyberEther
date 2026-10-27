#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/flowgraph.hh"
#include "compositor/default/actions/workbench.hh"
#include "flowgraph_fixture.hh"

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
    CHECK_FALSE(state.interface.focusedTab.has_value());

    actions.handle(MailOpenStartTab{});
    CHECK(state.startTabActive());

    state.interface.focusedTab.reset();
    actions.handle(MailFocusStartTab{});
    CHECK(state.startTabActive());

    actions.handle(MailCloseStartTab{});
    CHECK_FALSE(state.interface.focusedTab.has_value());

    actions.handle(MailOpenStartTab{});
    state.flowgraph.items["b"] = std::make_shared<Flowgraph>();
    actions.settleStartTab();
    CHECK_FALSE(state.interface.startTab);
    CHECK_FALSE(state.interface.focusedTab.has_value());
}

TEST_CASE("Workspace tabs suppress the active flowgraph without forgetting its selection",
          "[core][compositor][workspace]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    FlowgraphActions flowgraphs(state, callbacks);
    CHECK(state.tabsHidden());
    CHECK_FALSE(state.tabShown(WorkspaceTab::Start));
    CHECK_FALSE(state.activeFlowgraph().has_value());

    state.flowgraph.items["graph"] = std::make_shared<Flowgraph>();
    flowgraphs.handle(MailFocusFlowgraph{.flowgraph = "graph"});
    CHECK_FALSE(state.tabsHidden());
    CHECK(state.activeFlowgraph() == std::optional<std::string>{"graph"});

    for (const auto tab : {WorkspaceTab::Start, WorkspaceTab::Settings}) {
        WorkbenchActions::Open(state, tab);
        CHECK(state.tabShown(tab));
        CHECK(state.tabActive(tab));
        CHECK_FALSE(state.activeFlowgraph().has_value());
        CHECK(state.interface.focusedFlowgraph == std::optional<std::string>{"graph"});

        flowgraphs.handle(MailFocusFlowgraph{.flowgraph = "missing"});
        CHECK(state.tabActive(tab));
        flowgraphs.handle(MailFocusFlowgraph{.flowgraph = "graph"});
        CHECK_FALSE(state.interface.focusedTab.has_value());
        CHECK(state.tabShown(tab));
        CHECK(state.activeFlowgraph() == std::optional<std::string>{"graph"});
        WorkbenchActions::Close(state, tab);
    }

    state.flowgraph.items.clear();
    CHECK_FALSE(state.activeFlowgraph().has_value());
    CHECK(state.tabsHidden());
}

TEST_CASE("Hidden tab focus and background tab closure leave the active workspace tab alone",
          "[core][compositor][workspace]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    WorkbenchActions actions(state, callbacks);
    actions.handle(MailOpenSettings{});
    actions.handle(MailFocusStartTab{});
    CHECK(state.settingsTabActive());
    CHECK_FALSE(state.tabsHidden());

    actions.handle(MailOpenStartTab{});
    actions.handle(MailCloseSettings{});
    actions.handle(MailFocusSettings{});
    CHECK(state.startTabActive());
    CHECK_FALSE(state.settingsTabActive());

    actions.handle(MailOpenSettings{});
    state.flowgraph.items["new"] = std::make_shared<Flowgraph>();
    actions.settleStartTab();
    CHECK_FALSE(state.startTabShown());
    CHECK(state.settingsTabActive());
}

TEST_CASE("Flowgraph reconciliation preserves pending focus and clears stale floating views",
          "[core][compositor][workspace]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    WorkbenchActions actions(state, callbacks);
    state.flowgraph.items["graph"] = std::make_shared<Flowgraph>();
    state.interface.pendingFocusedFlowgraph = "pending";
    actions.reconcileFlowgraphs();
    CHECK_FALSE(state.interface.focusedFlowgraph.has_value());
    CHECK(state.interface.pendingFocusedFlowgraph == std::optional<std::string>{"pending"});

    state.interface.pendingFocusedFlowgraph.reset();
    actions.handle(MailOpenSettings{});
    actions.reconcileFlowgraphs();
    CHECK(state.interface.focusedFlowgraph == std::optional<std::string>{"graph"});
    CHECK(state.settingsTabActive());

    for (const bool empty : {false, true}) {
        if (empty) {
            state.flowgraph.items.clear();
        }
        state.interface.focusedFlowgraph = "removed";
        state.interface.flowgraphMetadataVisible = true;
        state.interface.flowgraphMetadataSearch = "metadata";
        state.interface.flowgraphEnvironmentVisible = true;
        state.interface.flowgraphEnvironmentSearch = "environment";
        actions.reconcileFlowgraphs();
        CHECK_FALSE(state.interface.flowgraphMetadataVisible);
        CHECK(state.interface.flowgraphMetadataSearch.empty());
        CHECK_FALSE(state.interface.flowgraphEnvironmentVisible);
        CHECK(state.interface.flowgraphEnvironmentSearch.empty());
        CHECK(state.interface.focusedFlowgraph == (empty ? std::nullopt : std::optional<std::string>{"graph"}));
        CHECK(state.settingsTabActive());
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Workbench reconciles legacy graph modals against their explicit or focused target",
                 "[core][compositor][workspace]") {
    REQUIRE(flowgraph->blockCreate("source", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    WorkbenchActions actions(state, callbacks);
    state.flowgraph.items["graph"] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});
    state.interface.focusedFlowgraph = "graph";

    for (const auto content : {ModalContent::FlowgraphInfo, ModalContent::FlowgraphClose, ModalContent::RenameBlock}) {
        state.modal.content = content;
        state.modal.flowgraph.reset();
        state.modal.renameBlockOldName = "source";
        actions.reconcileModal();
        CHECK(state.modal.content == content);

        state.modal.flowgraph = "removed";
        actions.reconcileModal();
        CHECK_FALSE(state.modal.content.has_value());
        CHECK_FALSE(state.modal.flowgraph.has_value());
    }

    state.modal.content = ModalContent::RenameBlock;
    state.modal.flowgraph = "graph";
    state.modal.renameBlockOldName = "removed";
    actions.reconcileModal();
    CHECK_FALSE(state.modal.content.has_value());
    CHECK_FALSE(state.modal.flowgraph.has_value());
    CHECK_FALSE(state.modal.renameBlockOldName.has_value());

    state.modal.flowgraph = "graph";
    actions.reconcileModal();
    CHECK_FALSE(state.modal.flowgraph.has_value());

    state.modal.content = ModalContent::RemoteStreaming;
    actions.reconcileModal();
    CHECK(state.modal.content == ModalContent::RemoteStreaming);
}
