#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/flowgraph.hh"
#include "compositor/default/actions/gallery.hh"
#include "compositor/default/actions/workbench.hh"
#include "environment.hh"
#include "flowgraph_fixture.hh"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

using namespace Jetstream;

TEST_CASE("Start tab opens once per request and closes when a flowgraph is added or the last one closes",
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

    state.flowgraph.items["b"] = std::make_shared<Flowgraph>();
    actions.settleStartTab();
    CHECK_FALSE(state.interface.startTab);

    actions.handle(MailOpenStartTab{});
    CHECK(state.interface.startTab);
    state.flowgraph.items.erase("a");
    actions.settleStartTab();
    CHECK(state.interface.startTab);
    state.flowgraph.items.erase("b");
    actions.settleStartTab();
    CHECK_FALSE(state.interface.startTab);
    CHECK(state.startTabActive());
    CHECK(state.startTabShown());
    CHECK(state.tabsHidden());
    state.flowgraph.items["a"] = std::make_shared<Flowgraph>();

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

TEST_CASE("Opening the start tab without flowgraphs takes focus from settings and docs",
          "[core][compositor][start-tab]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    WorkbenchActions actions(state, callbacks);
    actions.settleStartTab();
    CHECK(state.startTabActive());

    for (const auto tab : {WorkspaceTab::Settings, WorkspaceTab::Docs}) {
        WorkbenchActions::Open(state, tab);
        actions.settleStartTab();
        REQUIRE(state.tabActive(tab));

        const U64 requests = state.interface.startTabRequests;
        actions.handle(MailOpenStartTab{});
        actions.settleStartTab();
        CHECK_FALSE(state.interface.startTab);
        CHECK(state.startTabActive());
        CHECK_FALSE(state.tabActive(tab));
        CHECK(state.interface.startTabRequests == requests + 1);

        actions.handle(MailCloseStartTab{});
        actions.settleStartTab();
        CHECK(state.startTabActive());
    }
}

TEST_CASE("The start tab takes focus when the last flowgraph closes behind settings or docs",
          "[core][compositor][start-tab]") {
    for (const auto tab : {WorkspaceTab::Settings, WorkspaceTab::Docs}) {
        DefaultCompositorState state;
        DefaultCompositorCallbacks callbacks;
        WorkbenchActions actions(state, callbacks);
        state.flowgraph.items["a"] = std::make_shared<Flowgraph>();
        actions.settleStartTab();

        WorkbenchActions::Open(state, tab);
        actions.settleStartTab();
        REQUIRE(state.tabActive(tab));
        CHECK_FALSE(state.startTabShown());

        const U64 requests = state.interface.startTabRequests;
        state.flowgraph.items.erase("a");
        actions.reconcileFlowgraphs();
        actions.settleStartTab();
        CHECK(state.startTabActive());
        CHECK_FALSE(state.tabActive(tab));
        CHECK(state.tabShown(tab));
        CHECK_FALSE(state.activeFlowgraph().has_value());
        CHECK(state.interface.startTabRequests == requests + 1);
    }
}

TEST_CASE("The last tab navigated to in a poll wins over a newly shown start tab",
          "[core][compositor][start-tab]") {
    for (const auto tab : {WorkspaceTab::Settings, WorkspaceTab::Docs}) {
        DefaultCompositorState state;
        DefaultCompositorCallbacks callbacks;
        WorkbenchActions actions(state, callbacks);
        state.flowgraph.items["graph"] = std::make_shared<Flowgraph>();
        actions.settleStartTab();
        REQUIRE_FALSE(state.startTabShown());

        actions.handle(MailOpenStartTab{});
        WorkbenchActions::Open(state, tab);
        actions.settleStartTab();
        CHECK(state.startTabShown());
        CHECK(state.tabActive(tab));
        CHECK_FALSE(state.startTabActive());
    }
}

TEST_CASE("Favorite flowgraphs toggle from recent and outlive the recent list", "[core][compositor][start-tab]") {
    DefaultCompositorState state;
    auto& ui = state.interface;
    FlowgraphActions::Remember(ui, "/flows/radio.yml", "Radio", 10);
    FlowgraphActions::ToggleFavorite(ui, "/flows/radio.yml");
    REQUIRE(ui.favoriteFlowgraphs.size() == 1);
    CHECK(ui.favoriteFlowgraphs[0].title == "Radio");
    CHECK(ui.favoriteFlowgraphs[0].accessed == 10);

    for (U64 index = 0; index < DefaultCompositorState::InterfaceState::MaxRecentFlowgraphs; ++index) {
        FlowgraphActions::Remember(ui, "/flows/other" + std::to_string(index) + ".yml", "Other", 20);
    }
    CHECK(ui.recentFlowgraphs.size() == DefaultCompositorState::InterfaceState::MaxRecentFlowgraphs);
    CHECK(std::none_of(ui.recentFlowgraphs.begin(), ui.recentFlowgraphs.end(), [](const auto& entry) {
        return entry.path == "/flows/radio.yml";
    }));
    REQUIRE(ui.favoriteFlowgraphs.size() == 1);

    FlowgraphActions::Remember(ui, "/flows/radio.yml", "Radio Renamed", 30);
    CHECK(ui.favoriteFlowgraphs[0].title == "Radio Renamed");
    CHECK(ui.favoriteFlowgraphs[0].accessed == 30);

    FlowgraphActions::ToggleFavorite(ui, "/flows/radio.yml");
    CHECK(ui.favoriteFlowgraphs.empty());
}

TEST_CASE("Welcome section mail selects the explore page", "[core][compositor][start-tab]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    WorkbenchActions actions(state, callbacks);
    CHECK(state.interface.welcomeSection == WelcomeSection::All);
    actions.handle(MailSetWelcomeSection{.section = WelcomeSection::Flowgraphs});
    CHECK(state.interface.welcomeSection == WelcomeSection::Flowgraphs);
}

TEST_CASE("Welcome filter mails store the search text, category and page and reset the page on filter changes",
          "[core][compositor][start-tab]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    WorkbenchActions actions(state, callbacks);
    actions.handle(MailSetWelcomeQuery{.query = "radio"});
    actions.handle(MailSetWelcomeCategory{.category = "Receivers"});
    CHECK(state.interface.welcomeQuery == "radio");
    CHECK(state.interface.welcomeCategory == "Receivers");

    actions.handle(MailSetWelcomePage{.page = 2});
    CHECK(state.interface.welcomePage == 2);
    actions.handle(MailSetWelcomeQuery{.query = "fm"});
    CHECK(state.interface.welcomePage == 0);
    actions.handle(MailSetWelcomePage{.page = 1});
    actions.handle(MailSetWelcomeCategory{.category = "Basics"});
    CHECK(state.interface.welcomePage == 0);
    actions.handle(MailSetWelcomePage{.page = 1});
    actions.handle(MailSetWelcomeSection{.section = WelcomeSection::Flowgraphs});
    CHECK(state.interface.welcomePage == 0);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Closing a flowgraph with blocks asks first and cancelling or forcing settles the prompt",
                 "[core][compositor][flowgraph-actions]") {
    REQUIRE(flowgraph->blockCreate("source", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    U64 commands = 0;
    callbacks.enqueueCommand = [&](std::function<Result()>, bool) {
        ++commands;
    };
    FlowgraphActions actions(state, callbacks);
    state.flowgraph.items["graph"] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});
    state.flowgraph.items["empty"] = std::make_shared<Flowgraph>();

    REQUIRE(actions.handle(MailCloseFlowgraph{.flowgraph = "graph"}) == Result::SUCCESS);
    CHECK(state.interface.closingFlowgraph == std::optional<std::string>("graph"));
    CHECK(commands == 0);

    REQUIRE(actions.handle(MailCancelCloseFlowgraph{}) == Result::SUCCESS);
    CHECK_FALSE(state.interface.closingFlowgraph.has_value());

    REQUIRE(actions.handle(MailCloseFlowgraph{.flowgraph = "graph"}) == Result::SUCCESS);
    REQUIRE(actions.handle(MailCloseFlowgraph{.flowgraph = "graph", .force = true}) == Result::SUCCESS);
    CHECK_FALSE(state.interface.closingFlowgraph.has_value());
    CHECK(commands == 1);

    REQUIRE(actions.handle(MailCloseFlowgraph{.flowgraph = "empty"}) == Result::SUCCESS);
    CHECK_FALSE(state.interface.closingFlowgraph.has_value());
    CHECK(commands == 2);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Pausing a flowgraph from the editor toolbar pauses and resumes its scheduler",
                 "[core][compositor][flowgraph-actions]") {
    REQUIRE(flowgraph->blockCreate("source", TestFlowgraph::kSyntheticSourceType, {}, {}) == Result::SUCCESS);
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    FlowgraphActions actions(state, callbacks);
    state.flowgraph.items["graph"] = std::shared_ptr<Flowgraph>(flowgraph.get(), [](Flowgraph*) {});

    REQUIRE(actions.handle(MailSetFlowgraphPaused{.flowgraph = "missing", .paused = true}) == Result::SUCCESS);
    CHECK_FALSE(flowgraph->paused());
    REQUIRE(actions.handle(MailSetFlowgraphPaused{.flowgraph = "graph", .paused = true}) == Result::SUCCESS);
    CHECK(flowgraph->paused());
    REQUIRE(actions.handle(MailSetFlowgraphPaused{.flowgraph = "graph", .paused = false}) == Result::SUCCESS);
    CHECK_FALSE(flowgraph->paused());
}

TEST_CASE("Gallery page links open the matching example inside the app", "[core][compositor][start-tab]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    std::vector<Mail> mails;
    callbacks.enqueueMail = [&](Mail&& mail) {
        mails.push_back(std::move(mail));
    };
    GalleryActions(state).refreshRegistry();
    REQUIRE_FALSE(state.registry.flowgraphs.empty());
    const std::string key = state.registry.flowgraphs.front().key;
    WorkbenchActions actions(state, callbacks);

    REQUIRE(actions.handle(MailOpenUrl{.url = "https://cyberether.org/gallery/flowgraphs/" + key}) == Result::SUCCESS);
    REQUIRE(mails.size() == 1);
    const auto* open = std::get_if<MailOpenFlowgraphExample>(&mails.front());
    REQUIRE(open != nullptr);
    CHECK(open->key == key);

    auto renamed = state.registry.flowgraphs.front();
    renamed.key = std::string(GalleryActions::PublishedPrefix) + "demo";
    auto plugin = state.registry.flowgraphs.front();
    plugin.key = "demo";
    state.registry.flowgraphs = {plugin, renamed};
    mails.clear();
    REQUIRE(actions.handle(MailOpenUrl{.url = "https://cyberether.org/gallery/flowgraphs/demo"}) == Result::SUCCESS);
    REQUIRE(mails.size() == 1);
    open = std::get_if<MailOpenFlowgraphExample>(&mails.front());
    REQUIRE(open != nullptr);
    CHECK(open->key == "gallery/demo");
}

TEST_CASE("Docs page links open the bundled page inside the app", "[core][compositor][start-tab]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    std::vector<Mail> mails;
    callbacks.enqueueMail = [&](Mail&& mail) {
        mails.push_back(std::move(mail));
    };
    WorkbenchActions actions(state, callbacks);
    REQUIRE(Docs::Bundled().find("quick-start").has_value());

    REQUIRE(actions.handle(MailOpenUrl{.url = "https://cyberether.org/docs/quick-start#first-steps"}) ==
            Result::SUCCESS);
    REQUIRE(mails.size() == 1);
    const auto* open = std::get_if<MailOpenDocs>(&mails.front());
    REQUIRE(open != nullptr);
    CHECK(open->page == "quick-start");
    CHECK(open->anchor == "first-steps");
}

TEST_CASE("Gallery examples open from the cached registry by key", "[core][compositor][start-tab]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    U64 commands = 0;
    std::vector<std::function<void(Result)>> completions;
    std::string notice;
    callbacks.enqueueCommandWithCompletion = [&](std::function<Result()>, bool, std::function<void(Result)> done) {
        ++commands;
        completions.push_back(std::move(done));
    };
    callbacks.notify = [&](Sakura::ToastType, I32, const std::string& message) {
        notice = message;
    };
    GalleryActions(state).refreshRegistry();
    REQUIRE_FALSE(state.registry.flowgraphs.empty());
    FlowgraphActions actions(state, callbacks);

    REQUIRE(actions.handle(MailOpenFlowgraphExample{.key = "missing-example"}) == Result::SUCCESS);
    CHECK(commands == 0);
    CHECK_FALSE(notice.empty());
    CHECK_FALSE(state.interface.pendingFocusedFlowgraph.has_value());

    CHECK(state.flowgraph.gallery.empty());

    REQUIRE(actions.handle(MailOpenFlowgraphExample{.key = state.registry.flowgraphs.front().key}) == Result::SUCCESS);
    CHECK(commands == 1);
    REQUIRE(state.interface.pendingFocusedFlowgraph.has_value());
    REQUIRE(state.flowgraph.gallery.size() == 1);
    REQUIRE(state.flowgraph.gallery.contains(*state.interface.pendingFocusedFlowgraph));
    const auto& opened = state.registry.flowgraphs.front();
    CHECK(state.flowgraph.gallery.at(*state.interface.pendingFocusedFlowgraph) ==
          FlowgraphActions::GalleryDescription(opened));

    const auto& content = state.registry.flowgraphs.front().content;
    REQUIRE(actions.handle(MailOpenFlowgraphBlob{{content.begin(), content.end()}}) == Result::SUCCESS);
    CHECK(commands == 2);
    CHECK(state.flowgraph.gallery.size() == 1);
    CHECK_FALSE(state.flowgraph.gallery.contains(*state.interface.pendingFocusedFlowgraph));

    REQUIRE(completions.size() == 2);
    const auto blobName = *state.interface.pendingFocusedFlowgraph;
    completions[0](Result::ERROR);
    CHECK(state.flowgraph.gallery.empty());
    CHECK(state.interface.pendingFocusedFlowgraph == blobName);
    completions[1](Result::ERROR);
    CHECK_FALSE(state.interface.pendingFocusedFlowgraph.has_value());
}

TEST_CASE("Gallery and environment entries outlive a pending open and drop when their flowgraph closes",
          "[core][compositor][start-tab]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    WorkbenchActions actions(state, callbacks);

    state.flowgraph.gallery["first"] = "First";
    state.flowgraph.gallery["second"] = "Second";
    state.interface.pendingFocusedFlowgraph = "second";
    actions.reconcileFlowgraphs();
    CHECK(state.flowgraph.gallery.size() == 2);

    state.flowgraph.items["first"] = std::make_shared<Flowgraph>();
    state.flowgraph.environment["first"] = "";
    actions.reconcileFlowgraphs();
    CHECK(state.flowgraph.gallery.size() == 2);
    CHECK(state.flowgraph.environment.contains("first"));

    state.flowgraph.items.erase("first");
    actions.reconcileFlowgraphs();
    CHECK_FALSE(state.flowgraph.gallery.contains("first"));
    CHECK_FALSE(state.flowgraph.environment.contains("first"));
    CHECK(state.flowgraph.gallery.contains("second"));
}

TEST_CASE("Recent flowgraph history records a path only after its open or save succeeds",
          "[core][compositor][start-tab]") {
    TestSupport::SettingsSandbox sandbox("compositor-recent");
    sandbox.require();
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    std::vector<Mail> completions;
    std::vector<std::function<void(Result)>> imports;
    callbacks.enqueueCommandWithMail = [&](std::function<Result()>, bool, Mail mail) {
        completions.push_back(std::move(mail));
    };
    callbacks.enqueueCommandWithCompletion = [&](std::function<Result()>, bool, std::function<void(Result)> done) {
        imports.push_back(std::move(done));
    };
    callbacks.enqueueMail = [&](Mail&& mail) {
        completions.push_back(std::move(mail));
    };
    FlowgraphActions actions(state, callbacks);

    const auto opened = sandbox.root() / "opened.yml";
    std::filesystem::create_directories(sandbox.root());
    {
        std::ofstream file(opened);
        file << "title: Opened\n";
    }
    const auto openedPath = Platform::PathToUtf8(opened);
    REQUIRE(actions.handle(MailOpenFlowgraphPath{.path = openedPath}) == Result::SUCCESS);
    REQUIRE(actions.handle(MailOpenFlowgraphPath{.path = openedPath}) == Result::SUCCESS);
    REQUIRE(imports.size() == 2);
    CHECK(completions.empty());
    imports[0](Result::ERROR);
    CHECK(completions.empty());
    imports[1](Result::SUCCESS);
    CHECK(state.interface.recentFlowgraphs.empty());

    auto graph = std::make_shared<Flowgraph>();
    state.flowgraph.items["graph"] = graph;
    const auto savedPath = Platform::PathToUtf8(sandbox.root() / "saved.yml");
    REQUIRE(actions.handle(MailSaveFlowgraph{.flowgraph = "graph", .path = savedPath}) == Result::SUCCESS);
    CHECK(state.interface.recentFlowgraphs.empty());

    REQUIRE(completions.size() == 2);
    for (const auto& mail : completions) {
        REQUIRE(std::holds_alternative<MailRememberFlowgraph>(mail));
        REQUIRE(actions.handle(std::get<MailRememberFlowgraph>(mail)) == Result::SUCCESS);
    }

    const auto& recent = state.interface.recentFlowgraphs;
    REQUIRE(recent.size() == 2);
    CHECK(recent[0].path == savedPath);
    CHECK(recent[1].path == openedPath);
    CHECK(recent[1].title.empty());

    REQUIRE(Settings::Flush() == Result::SUCCESS);
    Settings settings;
    REQUIRE(Settings::Get(settings) == Result::SUCCESS);
    CHECK(settings.interface.recentFlowgraphs.size() == 2);
}
