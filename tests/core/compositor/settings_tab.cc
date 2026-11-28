#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/gallery.hh"
#include "compositor/default/actions/settings.hh"
#include "compositor/default/presenters/agent/labels.hh"
#include "compositor/default/presenters/settings/gpu_labels.hh"
#include "compositor/default/actions/workbench.hh"
#include "environment.hh"

#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

using namespace Jetstream;

TEST_CASE("Settings tab opens on the requested section and closes", "[core][compositor][settings-tab]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    WorkbenchActions actions(state, callbacks);

    actions.handle(MailOpenSettings{.section = SettingsSection::Remote});
    CHECK(state.interface.settingsTab);
    CHECK(state.settingsTabActive());
    CHECK(state.interface.settingsTabRequests == 1);
    CHECK(state.settings.section == SettingsSection::Remote);

    actions.handle(MailOpenSettings{.section = SettingsSection::About});
    CHECK(state.interface.settingsTabRequests == 2);
    CHECK(state.settings.section == SettingsSection::About);

    actions.handle(MailCloseSettings{});
    CHECK_FALSE(state.interface.settingsTab);
    CHECK_FALSE(state.settingsTabActive());
    CHECK(state.settings.section == SettingsSection::About);
}

TEST_CASE("Settings tab and start tab hand focus to each other", "[core][compositor][settings-tab]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    WorkbenchActions actions(state, callbacks);

    actions.handle(MailOpenStartTab{});
    actions.handle(MailOpenSettings{});
    CHECK(state.settingsTabActive());
    CHECK_FALSE(state.startTabActive());

    actions.handle(MailFocusStartTab{});
    CHECK(state.startTabActive());
    CHECK_FALSE(state.settingsTabActive());

    actions.handle(MailFocusSettings{});
    CHECK(state.settingsTabActive());
    CHECK_FALSE(state.startTabActive());

    actions.handle(MailCloseSettings{});
    actions.handle(MailFocusSettings{});
    CHECK_FALSE(state.settingsTabActive());
}

TEST_CASE("Registry pages navigate and reset when settings change section", "[core][compositor][settings-tab]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    WorkbenchActions workbench(state, callbacks);
    SettingsActions settings(state, callbacks);

    settings.handle(MailSetRegistryPage{.page = RegistryPage::Plugin, .plugin = "alpha"});
    CHECK(state.settings.registryPage == RegistryPage::Plugin);
    CHECK(state.settings.registryPlugin == "alpha");

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
    CHECK(state.settingsTabActive());
    state.interface.pendingFocusedFlowgraph = "fresh";
    actions.settlePendingFocus();
    CHECK(state.settingsTabActive());

    state.flowgraph.items["fresh"] = std::make_shared<Flowgraph>();
    actions.settlePendingFocus();
    CHECK(state.interface.focusedFlowgraph == std::optional<std::string>{"fresh"});
    CHECK_FALSE(state.interface.pendingFocusedFlowgraph.has_value());
    CHECK_FALSE(state.settingsTabActive());
    CHECK_FALSE(state.startTabActive());
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

TEST_CASE("Show tips updates the interface state and persists the setting", "[core][compositor][settings-tab]") {
    TestSupport::SettingsSandbox sandbox("compositor-settings");
    sandbox.require();
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    SettingsActions actions(state, callbacks);

    const auto stored = [] {
        REQUIRE(Settings::Flush() == Result::SUCCESS);
        std::string configPath;
        REQUIRE(Platform::ConfigPath(configPath) == Result::SUCCESS);
        std::ifstream file(Platform::PathFromUtf8(configPath) / "settings.yaml", std::ios::binary);
        REQUIRE(file);
        const std::string yaml((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        Parser::Map document;
        REQUIRE(Parser::YamlDecode(yaml, document) == Result::SUCCESS);
        Settings settings;
        REQUIRE(settings.deserialize(document) == Result::SUCCESS);
        return settings.interface.showTips;
    };

    REQUIRE(actions.handle(MailSetShowTips{.value = false}) == Result::SUCCESS);
    CHECK_FALSE(state.interface.showTips);
    CHECK_FALSE(stored());
    Settings settings;
    REQUIRE(Settings::Get(settings) == Result::SUCCESS);
    CHECK_FALSE(settings.interface.showTips);

    REQUIRE(actions.handle(MailSetShowTips{.value = true}) == Result::SUCCESS);
    CHECK(state.interface.showTips);
    CHECK(stored());
    REQUIRE(Settings::Get(settings) == Result::SUCCESS);
    CHECK(settings.interface.showTips);
}

TEST_CASE("Touchpad scroll speed updates the interface state and persists the setting", "[core][compositor][settings-tab]") {
    TestSupport::SettingsSandbox sandbox("compositor-scroll-speed");
    sandbox.require();
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    SettingsActions actions(state, callbacks);
    CHECK(state.interface.touchpadScrollSpeed == Settings::Interface::DefaultTouchpadScrollSpeed);

    REQUIRE(actions.handle(MailSetTouchpadScrollSpeed{.value = 2.4f}) == Result::SUCCESS);
    CHECK(state.interface.touchpadScrollSpeed == 2.4f);
    REQUIRE(Settings::Flush() == Result::SUCCESS);
    Settings settings;
    REQUIRE(Settings::Get(settings) == Result::SUCCESS);
    CHECK(settings.interface.touchpadScrollSpeed == 2.4f);
}

TEST_CASE("Selecting a GPU updates the graphics state and persists the UUID and name",
          "[core][compositor][settings-tab][gpu]") {
    TestSupport::SettingsSandbox sandbox("compositor-gpu");
    sandbox.require();
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    SettingsActions actions(state, callbacks);
    const Settings::Gpu nvidia = {.uuid = "nvidia-uuid", .name = "NVIDIA GeForce RTX 5060 Ti"};

    REQUIRE(actions.handle(MailSetGraphicsGpu{.value = nvidia}) == Result::SUCCESS);
    CHECK(state.graphics.gpu == nvidia);
    REQUIRE(Settings::Flush() == Result::SUCCESS);
    Settings settings;
    REQUIRE(Settings::Get(settings) == Result::SUCCESS);
    CHECK(settings.graphics.gpu == nvidia);
    CHECK_FALSE(settings.graphics.deviceId.has_value());

    REQUIRE(actions.handle(MailSetGraphicsGpu{}) == Result::SUCCESS);
    CHECK(state.graphics.gpu == Settings::Gpu{});
    REQUIRE(Settings::Flush() == Result::SUCCESS);
    REQUIRE(Settings::Get(settings) == Result::SUCCESS);
    CHECK(settings.graphics.gpu.uuid.empty());
}

namespace {

std::vector<Backend::PhysicalDevice> HybridLaptop() {
    return {
        {.index = 0, .name = "Intel(R) Graphics (PTL)", .uuid = "intel", .type = Backend::PhysicalDeviceType::INTEGRATED},
        {.index = 1, .name = "NVIDIA GeForce RTX 5060 Ti", .uuid = "nvidia", .type = Backend::PhysicalDeviceType::DISCRETE},
        {.index = 2, .name = "llvmpipe (LLVM 19.1.7, 256 bits)", .uuid = "cpu", .type = Backend::PhysicalDeviceType::CPU},
    };
}

std::vector<std::string> Labels(const GpuLabels::Choices& choices) {
    std::vector<std::string> labels;
    for (const auto& option : choices.options) {
        labels.push_back(option.label);
    }
    return labels;
}

std::vector<std::string> Details(const GpuLabels::Choices& choices) {
    std::vector<std::string> details;
    for (const auto& option : choices.options) {
        details.push_back(option.detail);
    }
    return details;
}

}  // namespace

TEST_CASE("GPU choices lead with what Automatic resolves to and hide CPU renderers",
          "[core][compositor][settings-tab][gpu]") {
    const auto choices = GpuLabels::Build(HybridLaptop(), {});
    CHECK(Labels(choices) == std::vector<std::string>{
        "Automatic (Intel(R) Graphics (PTL))",
        "Intel(R) Graphics (PTL)",
        "NVIDIA GeForce RTX 5060 Ti",
    });
    CHECK(Details(choices) == std::vector<std::string>{"", "Integrated", "Discrete"});
    CHECK(choices.selected == 0);
    CHECK_FALSE(choices.disabled);
    CHECK(choices.options[0].value == Settings::Gpu{});
    CHECK(choices.options[2].value == Settings::Gpu{.uuid = "nvidia", .name = "NVIDIA GeForce RTX 5060 Ti"});

    const auto saved = GpuLabels::Build(HybridLaptop(), {.uuid = "nvidia", .name = "NVIDIA GeForce RTX 5060 Ti"});
    CHECK(saved.selected == 2);
}

TEST_CASE("GPU choices keep a saved GPU that is not connected selected",
          "[core][compositor][settings-tab][gpu]") {
    const auto choices = GpuLabels::Build(HybridLaptop(), {.uuid = "egpu", .name = "AMD Radeon RX 7900 XTX"});
    REQUIRE(choices.options.size() == 4);
    CHECK(choices.options.back().label == "AMD Radeon RX 7900 XTX (not connected)");
    CHECK(choices.options.back().detail.empty());
    CHECK(choices.options.back().value == Settings::Gpu{.uuid = "egpu", .name = "AMD Radeon RX 7900 XTX"});
    CHECK(choices.selected == 3);
    CHECK_FALSE(choices.disabled);
}

TEST_CASE("GPU choices disable a single GPU and show CPU renderers only when nothing else exists",
          "[core][compositor][settings-tab][gpu]") {
    auto single = HybridLaptop();
    single.erase(single.begin() + 1);
    const auto choices = GpuLabels::Build(single, {});
    CHECK(Labels(choices) == std::vector<std::string>{"Automatic (Intel(R) Graphics (PTL))", "Intel(R) Graphics (PTL)"});
    CHECK(choices.disabled);
    const auto pinned = GpuLabels::Build(single, {.uuid = "intel", .name = "Intel(R) Graphics (PTL)"});
    CHECK(pinned.selected == 1);
    CHECK_FALSE(pinned.disabled);

    const std::vector<Backend::PhysicalDevice> software = {
        {.index = 0, .name = "llvmpipe", .uuid = "cpu", .type = Backend::PhysicalDeviceType::CPU},
    };
    const auto cpu = GpuLabels::Build(software, {});
    CHECK(Labels(cpu) == std::vector<std::string>{"Automatic (llvmpipe)", "llvmpipe"});
    CHECK(Details(cpu) == std::vector<std::string>{"", "CPU"});
    CHECK(cpu.disabled);

    const std::vector<Backend::PhysicalDevice> softwareFirst = {
        {.index = 0, .name = "llvmpipe", .uuid = "cpu", .type = Backend::PhysicalDeviceType::CPU},
        {.index = 1, .name = "NVIDIA GeForce RTX 5060 Ti", .uuid = "nvidia", .type = Backend::PhysicalDeviceType::DISCRETE},
    };
    const auto reachable = GpuLabels::Build(softwareFirst, {});
    CHECK(Labels(reachable) == std::vector<std::string>{"Automatic (llvmpipe)", "NVIDIA GeForce RTX 5060 Ti"});
    CHECK_FALSE(reachable.disabled);

    const auto none = GpuLabels::Build({}, {});
    CHECK(Labels(none) == std::vector<std::string>{"Automatic"});
    CHECK(none.disabled);

    const auto offline = GpuLabels::Build({}, {.uuid = "nvidia", .name = "NVIDIA GeForce RTX 5060 Ti"});
    CHECK(Labels(offline) == std::vector<std::string>{"Automatic", "NVIDIA GeForce RTX 5060 Ti (not connected)"});
    CHECK(offline.selected == 1);
    CHECK_FALSE(offline.disabled);
}

TEST_CASE("GPU names number duplicates by PCI slot", "[core][compositor][settings-tab][gpu]") {
    const std::vector<Backend::PhysicalDevice> devices = {
        {.index = 0, .name = "NVIDIA RTX A4000", .uuid = "b", .type = Backend::PhysicalDeviceType::DISCRETE,
         .pciAddress = "0000:41:00.0"},
        {.index = 1, .name = "NVIDIA RTX A4000", .uuid = "a", .type = Backend::PhysicalDeviceType::DISCRETE,
         .pciAddress = "0000:01:00.0"},
        {.index = 2, .name = "AMD Radeon RX 7900 XTX", .uuid = "c", .type = Backend::PhysicalDeviceType::EXTERNAL,
         .pciAddress = "0000:0a:00.0"},
    };
    CHECK(GpuLabels::Names(devices) == std::vector<std::string>{
        "NVIDIA RTX A4000 #2",
        "NVIDIA RTX A4000 #1",
        "AMD Radeon RX 7900 XTX",
    });
    const auto choices = GpuLabels::Build(devices, {});
    CHECK(choices.options[0].label == "Automatic (NVIDIA RTX A4000 #2)");
    CHECK(Details(choices) == std::vector<std::string>{"", "Discrete", "Discrete", "External"});
    CHECK(GpuLabels::Type(Backend::PhysicalDeviceType::VIRTUAL) == "Virtual");
}

TEST_CASE("Adding a plugin path that fails validation leaves the settings tab and registry snapshot alone",
          "[core][compositor][settings-tab]") {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    std::string notice;
    callbacks.notify = [&](Sakura::ToastType, I32, const std::string& message) {
        notice = message;
    };
    TestSupport::SettingsSandbox sandbox("compositor-settings");
    sandbox.require();
    Home::Gallery gallery;
    GalleryActions(state, gallery).refreshRegistry();
    SettingsActions actions(state, callbacks);

    REQUIRE(actions.handle(MailInstallPlugin{.path = "/plugins/alpha.txt"}) == Result::SUCCESS);
    CHECK(notice == "Plugins must be .cep bundles.");
    CHECK_FALSE(state.interface.settingsTab);
    CHECK(state.interface.settingsTabRequests == 0);
    CHECK_FALSE(state.registry.stale);
}
