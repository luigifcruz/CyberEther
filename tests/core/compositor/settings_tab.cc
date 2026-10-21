#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/settings.hh"
#include "compositor/default/presenters/agent/labels.hh"
#include "compositor/default/actions/workbench.hh"

#include <memory>

using namespace Jetstream;

TEST_CASE("Settings tab opens on the requested section and closes", "[core][compositor][settings-tab]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    WorkbenchActions actions(state, callbacks);

    actions.handle(MailOpenSettings{.section = SettingsSection::Remote});
    CHECK(state.interface.settingsTab);
    CHECK(state.interface.settingsTabFocused);
    CHECK(state.interface.settingsTabRequests == 1);
    CHECK(state.settings.section == SettingsSection::Remote);

    actions.handle(MailOpenSettings{.section = SettingsSection::About});
    CHECK(state.interface.settingsTabRequests == 2);
    CHECK(state.settings.section == SettingsSection::About);

    actions.handle(MailCloseSettings{});
    CHECK_FALSE(state.interface.settingsTab);
    CHECK_FALSE(state.interface.settingsTabFocused);
    CHECK(state.settings.section == SettingsSection::About);
}

TEST_CASE("Settings tab and start tab hand focus to each other", "[core][compositor][settings-tab]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    WorkbenchActions actions(state, callbacks);

    actions.handle(MailOpenStartTab{});
    actions.handle(MailOpenSettings{});
    CHECK(state.interface.settingsTabFocused);
    CHECK_FALSE(state.interface.startTabFocused);

    actions.handle(MailFocusStartTab{});
    CHECK(state.interface.startTabFocused);
    CHECK_FALSE(state.interface.settingsTabFocused);

    actions.handle(MailFocusSettings{});
    CHECK(state.interface.settingsTabFocused);
    CHECK_FALSE(state.interface.startTabFocused);

    actions.handle(MailCloseSettings{});
    actions.handle(MailFocusSettings{});
    CHECK_FALSE(state.interface.settingsTabFocused);
}

TEST_CASE("Registry pages navigate and reset when settings change section", "[core][compositor][settings-tab]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    WorkbenchActions workbench(state, callbacks);
    SettingsActions settings(state, callbacks);

    settings.handle(MailSetRegistryPage{.page = RegistryPage::Plugin, .plugin = "/plugins/alpha.cep"});
    CHECK(state.settings.registryPage == RegistryPage::Plugin);
    CHECK(state.settings.registryPlugin == "/plugins/alpha.cep");

    settings.handle(MailSetSettingsSection{.section = SettingsSection::General});
    CHECK(state.settings.registryPage == RegistryPage::Overview);
    CHECK(state.settings.registryPlugin.empty());

    settings.handle(MailSetRegistryPage{.page = RegistryPage::AddPlugin});
    workbench.handle(MailOpenSettings{.section = SettingsSection::Registry});
    CHECK(state.settings.registryPage == RegistryPage::Overview);
    CHECK(state.settings.section == SettingsSection::Registry);
}

TEST_CASE("Intelligence pages navigate and reset when settings change section", "[core][compositor][settings-tab]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    SettingsActions settings(state, callbacks);

    settings.handle(MailSetIntelligencePage{.page = IntelligencePage::AddModel});
    CHECK(state.settings.intelligencePage == IntelligencePage::AddModel);
    settings.handle(MailSetSettingsSection{.section = SettingsSection::Intelligence});
    CHECK(state.settings.intelligencePage == IntelligencePage::Overview);
}

TEST_CASE("Agent labels format prices and voice identifiers for display", "[core][compositor][settings-tab]") {
    CHECK(AgentLabels::Price(3.0) == "$3.00");
    CHECK(AgentLabels::Price(0.6) == "$0.60");
    CHECK(AgentLabels::Price(1.25) == "$1.25");
    CHECK(AgentLabels::Price(0.15) == "$0.15");
    CHECK(AgentLabels::Price(0.075) == "$0.075");
    CHECK(AgentLabels::Price(0.0) == "Free");
    CHECK(AgentLabels::Title("gpt-live-1") == "GPT Live 1");
    CHECK(AgentLabels::Title("marin") == "Marin");
}

TEST_CASE("A newly focused flowgraph takes focus away from the settings and start tabs",
          "[core][compositor][settings-tab]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    WorkbenchActions actions(state, callbacks);

    actions.handle(MailOpenSettings{});
    CHECK(state.interface.settingsTabFocused);
    state.interface.pendingFocusedFlowgraph = "fresh";
    actions.settlePendingFocus();
    CHECK(state.interface.settingsTabFocused);

    state.flowgraph.items["fresh"] = std::make_shared<Flowgraph>();
    actions.settlePendingFocus();
    CHECK(state.interface.focusedFlowgraph == std::optional<std::string>{"fresh"});
    CHECK_FALSE(state.interface.pendingFocusedFlowgraph.has_value());
    CHECK_FALSE(state.interface.settingsTabFocused);
    CHECK_FALSE(state.interface.startTabFocused);
}

TEST_CASE("Closing settings resets its sub-pages so late saves do not reopen it", "[core][compositor][settings-tab]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    WorkbenchActions workbench(state, callbacks);
    SettingsActions settings(state, callbacks);

    workbench.handle(MailOpenSettings{.section = SettingsSection::Intelligence});
    settings.handle(MailSetIntelligencePage{.page = IntelligencePage::Provider});
    settings.handle(MailSetRegistryPage{.page = RegistryPage::AddPlugin});
    workbench.handle(MailCloseSettings{});
    CHECK_FALSE(state.interface.settingsTab);
    CHECK(state.settings.intelligencePage == IntelligencePage::Overview);
    CHECK(state.settings.registryPage == RegistryPage::Overview);
}
