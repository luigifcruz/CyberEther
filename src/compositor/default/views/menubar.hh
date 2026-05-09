#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MENUBAR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MENUBAR_HH

#include "jetstream/render/sakura/sakura.hh"

#include <array>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct MenubarView : public Sakura::Component {
    enum class Action {
        About,
        ViewLicense,
        ViewThirdPartyOss,
        CheckForUpdates,
        Preferences,
        Quit,
        NewFlowgraph,
        OpenFlowgraph,
        SaveFlowgraph,
        ShowFlowgraphInfo,
        CloseFlowgraph,
        RenameFlowgraph,
        OpenExamples,
        ToggleInfoPanel,
        ToggleBackgroundParticles,
        RemoteStreaming,
        OpenSettings,
        ToggleDebugLatency,
        ToggleRuntimeMetrics,
        ShowBenchmarks,
        OpenDeveloperSettings,
        GettingStarted,
        OpenTwitter,
        Documentation,
        OpenRepository,
        ReportIssue,
    };

    struct Config {
        std::string id;
        bool hasFocusedFlowgraph = false;
        bool infoPanelEnabled = false;
        bool backgroundParticles = false;
        bool remoteSupported = false;
        bool debugLatencyEnabled = false;
        bool debugRuntimeMetricsEnabled = false;
        I32 debugLogLevel = 0;
        std::vector<std::string> themes;
        std::string currentThemeKey;
        std::function<void(F32)> onHeight;
        std::function<void(Action)> onAction;
        std::function<void(const std::string&)> onThemeSelect;
        std::function<void(I32)> onDebugLogLevelSelect;
    };

    void update(Config config) {
        this->config = std::move(config);
        menuBar.update({
            .id = this->config.id + ":bar",
            .heightScale = 1.5f,
            .onHeight = this->config.onHeight,
        });

        appMenu.update({
            .id = this->config.id + ":app",
            .label = "CyberEther",
            .font = Sakura::Text::Font::Bold,
            .scale = 1.5f,
            .colorKey = "cyber_blue",
        });
        aboutItem.update({
            .id = this->config.id + ":about",
            .label = "About CyberEther",
            .onClick = [this]() { emit(Action::About); },
        });
        licenseItem.update({
            .id = this->config.id + ":license",
            .label = "View License",
            .onClick = [this]() { emit(Action::ViewLicense); },
        });
        thirdPartyItem.update({
            .id = this->config.id + ":third-party",
            .label = "Third-Party OSS",
            .onClick = [this]() { emit(Action::ViewThirdPartyOss); },
        });
        checkUpdatesItem.update({
            .id = this->config.id + ":check-updates",
            .label = "Check for Updates",
            .onClick = [this]() { emit(Action::CheckForUpdates); },
        });
        preferencesItem.update({
            .id = this->config.id + ":preferences",
            .label = "Preferences...",
            .shortcut = "CTRL+,",
            .onClick = [this]() { emit(Action::Preferences); },
        });
        quitItem.update({
            .id = this->config.id + ":quit",
            .label = "Quit CyberEther",
            .onClick = [this]() { emit(Action::Quit); },
        });

        flowgraphMenu.update({
            .id = this->config.id + ":flowgraph",
            .label = "Flowgraph",
            .scale = 1.04f,
        });
        newFlowgraphItem.update({
            .id = this->config.id + ":new-flowgraph",
            .label = "New",
            .shortcut = "CTRL+N",
            .onClick = [this]() { emit(Action::NewFlowgraph); },
        });
        openFlowgraphItem.update({
            .id = this->config.id + ":open-flowgraph",
            .label = "Open",
            .shortcut = "CTRL+O",
            .onClick = [this]() { emit(Action::OpenFlowgraph); },
        });
        saveFlowgraphItem.update({
            .id = this->config.id + ":save-flowgraph",
            .label = "Save",
            .shortcut = "CTRL+S",
            .enabled = this->config.hasFocusedFlowgraph,
            .onClick = [this]() { emit(Action::SaveFlowgraph); },
        });
        flowgraphInfoItem.update({
            .id = this->config.id + ":flowgraph-info",
            .label = "Info",
            .shortcut = "CTRL+I",
            .enabled = this->config.hasFocusedFlowgraph,
            .onClick = [this]() { emit(Action::ShowFlowgraphInfo); },
        });
        closeFlowgraphItem.update({
            .id = this->config.id + ":close-flowgraph",
            .label = "Close",
            .shortcut = "CTRL+W",
            .enabled = this->config.hasFocusedFlowgraph,
            .onClick = [this]() { emit(Action::CloseFlowgraph); },
        });
        renameFlowgraphItem.update({
            .id = this->config.id + ":rename-flowgraph",
            .label = "Rename",
            .enabled = this->config.hasFocusedFlowgraph,
            .onClick = [this]() { emit(Action::RenameFlowgraph); },
        });
        openExamplesItem.update({
            .id = this->config.id + ":open-examples",
            .label = "Open Examples",
            .onClick = [this]() { emit(Action::OpenExamples); },
        });

        viewMenu.update({
            .id = this->config.id + ":view",
            .label = "View",
            .scale = 1.04f,
        });
        infoPanelItem.update({
            .id = this->config.id + ":info-panel",
            .label = "Show Info Panel",
            .selected = this->config.infoPanelEnabled,
            .onClick = [this]() { emit(Action::ToggleInfoPanel); },
        });
        backgroundParticlesItem.update({
            .id = this->config.id + ":background-particles",
            .label = "Background Particles",
            .selected = this->config.backgroundParticles,
            .onClick = [this]() { emit(Action::ToggleBackgroundParticles); },
        });
        themeMenu.update({
            .id = this->config.id + ":theme",
            .label = "Theme",
        });
        themeItems.resize(this->config.themes.size());
        for (U64 i = 0; i < themeItems.size(); ++i) {
            const auto& theme = this->config.themes[i];
            themeItems[i].update({
                .id = this->config.id + ":theme:" + theme,
                .label = theme,
                .selected = this->config.currentThemeKey == theme,
                .onClick = [this, theme]() {
                    if (this->config.onThemeSelect) {
                        this->config.onThemeSelect(theme);
                    }
                },
            });
        }
        remoteStreamingItem.update({
            .id = this->config.id + ":remote-streaming",
            .label = "Remote Streaming",
            .enabled = this->config.remoteSupported,
            .onClick = [this]() { emit(Action::RemoteStreaming); },
        });
        openSettingsItem.update({
            .id = this->config.id + ":open-settings",
            .label = "Open In Settings",
            .onClick = [this]() { emit(Action::OpenSettings); },
        });

        developerMenu.update({
            .id = this->config.id + ":developer",
            .label = "Developer",
            .scale = 1.04f,
        });
        latencyWindowItem.update({
            .id = this->config.id + ":latency-window",
            .label = "Show Latency Window",
            .selected = this->config.debugLatencyEnabled,
            .onClick = [this]() { emit(Action::ToggleDebugLatency); },
        });
        runtimeMetricsItem.update({
            .id = this->config.id + ":runtime-metrics",
            .label = "Show Runtime Metrics",
            .selected = this->config.debugRuntimeMetricsEnabled,
            .onClick = [this]() { emit(Action::ToggleRuntimeMetrics); },
        });
        benchmarksItem.update({
            .id = this->config.id + ":benchmarks",
            .label = "Show Benchmarks",
            .onClick = [this]() { emit(Action::ShowBenchmarks); },
        });
        logLevelMenu.update({
            .id = this->config.id + ":log-level",
            .label = "Log Level",
        });
        logLevelItems.resize(logLevelOptions.size());
        for (U64 i = 0; i < logLevelItems.size(); ++i) {
            logLevelItems[i].update({
                .id = this->config.id + ":log-level:" + std::to_string(i),
                .label = logLevelOptions[i],
                .selected = this->config.debugLogLevel == static_cast<I32>(i),
                .onClick = [this, i]() {
                    if (this->config.onDebugLogLevelSelect) {
                        this->config.onDebugLogLevelSelect(static_cast<I32>(i));
                    }
                },
            });
        }
        openDeveloperSettingsItem.update({
            .id = this->config.id + ":open-developer-settings",
            .label = "Open In Settings",
            .onClick = [this]() { emit(Action::OpenDeveloperSettings); },
        });

        helpMenu.update({
            .id = this->config.id + ":help",
            .label = "Help",
            .scale = 1.04f,
        });
        gettingStartedItem.update({
            .id = this->config.id + ":getting-started",
            .label = "Getting Started",
            .onClick = [this]() { emit(Action::GettingStarted); },
        });
        twitterItem.update({
            .id = this->config.id + ":twitter",
            .label = "Luigi's Twitter",
            .onClick = [this]() { emit(Action::OpenTwitter); },
        });
        documentationItem.update({
            .id = this->config.id + ":documentation",
            .label = "Documentation",
            .onClick = [this]() { emit(Action::Documentation); },
        });
        repositoryItem.update({
            .id = this->config.id + ":repository",
            .label = "Open Repository",
            .onClick = [this]() { emit(Action::OpenRepository); },
        });
        reportIssueItem.update({
            .id = this->config.id + ":report-issue",
            .label = "Report Issue",
            .onClick = [this]() { emit(Action::ReportIssue); },
        });

        for (U64 i = 0; i < dividers.size(); ++i) {
            dividers[i].update({
                .id = this->config.id + ":divider:" + std::to_string(i),
                .spacing = 0.0f,
            });
        }

        shortcuts.update({
            .id = this->config.id + ":shortcuts",
            .bindings = {
                {
                    .key = Sakura::KeyboardInput::Key::N,
                    .modifier = Sakura::KeyboardInput::Modifier::CommandOrControl,
                    .onPressed = [this]() { emit(Action::NewFlowgraph); },
                },
                {
                    .key = Sakura::KeyboardInput::Key::O,
                    .modifier = Sakura::KeyboardInput::Modifier::CommandOrControl,
                    .onPressed = [this]() { emit(Action::OpenFlowgraph); },
                },
                {
                    .key = Sakura::KeyboardInput::Key::S,
                    .modifier = Sakura::KeyboardInput::Modifier::CommandOrControl,
                    .onPressed = [this]() { emit(Action::SaveFlowgraph); },
                },
                {
                    .key = Sakura::KeyboardInput::Key::W,
                    .modifier = Sakura::KeyboardInput::Modifier::CommandOrControl,
                    .onPressed = [this]() { emit(Action::CloseFlowgraph); },
                },
                {
                    .key = Sakura::KeyboardInput::Key::I,
                    .modifier = Sakura::KeyboardInput::Modifier::CommandOrControl,
                    .onPressed = [this]() { emit(Action::ShowFlowgraphInfo); },
                },
                {
                    .key = Sakura::KeyboardInput::Key::Comma,
                    .modifier = Sakura::KeyboardInput::Modifier::CommandOrControl,
                    .onPressed = [this]() { emit(Action::Preferences); },
                },
            },
        });
    }

    void render(const Sakura::Context& ctx) {
        shortcuts.render(ctx);
        menuBar.render(ctx, [this](const Sakura::Context& ctx) {
            appMenu.render(ctx, [this](const Sakura::Context& ctx) {
                aboutItem.render(ctx);
                licenseItem.render(ctx);
                thirdPartyItem.render(ctx);
                dividers[0].render(ctx);
                checkUpdatesItem.render(ctx);
                dividers[1].render(ctx);
                preferencesItem.render(ctx);
#ifndef JST_OS_BROWSER
                dividers[2].render(ctx);
                quitItem.render(ctx);
#endif
            });

            flowgraphMenu.render(ctx, [this](const Sakura::Context& ctx) {
                newFlowgraphItem.render(ctx);
                openFlowgraphItem.render(ctx);
                saveFlowgraphItem.render(ctx);
                flowgraphInfoItem.render(ctx);
                closeFlowgraphItem.render(ctx);
                renameFlowgraphItem.render(ctx);
                dividers[3].render(ctx);
                openExamplesItem.render(ctx);
            });

            viewMenu.render(ctx, [this](const Sakura::Context& ctx) {
                infoPanelItem.render(ctx);
                backgroundParticlesItem.render(ctx);
                dividers[4].render(ctx);
                themeMenu.render(ctx, [this](const Sakura::Context& ctx) {
                    for (auto& item : themeItems) {
                        item.render(ctx);
                    }
                });
                dividers[5].render(ctx);
                remoteStreamingItem.render(ctx);
                dividers[6].render(ctx);
                openSettingsItem.render(ctx);
            });

            developerMenu.render(ctx, [this](const Sakura::Context& ctx) {
                latencyWindowItem.render(ctx);
                runtimeMetricsItem.render(ctx);
                benchmarksItem.render(ctx);
                dividers[7].render(ctx);
                logLevelMenu.render(ctx, [this](const Sakura::Context& ctx) {
                    for (auto& item : logLevelItems) {
                        item.render(ctx);
                    }
                });
                dividers[8].render(ctx);
                openDeveloperSettingsItem.render(ctx);
            });

            helpMenu.render(ctx, [this](const Sakura::Context& ctx) {
                gettingStartedItem.render(ctx);
                twitterItem.render(ctx);
                documentationItem.render(ctx);
                repositoryItem.render(ctx);
                reportIssueItem.render(ctx);
            });
        });
    }

 private:
    void emit(const Action action) const {
        if (config.onAction) {
            config.onAction(action);
        }
    }

    static constexpr std::array<const char*, 5> logLevelOptions = {
        "Fatal / Error",
        "Warnings",
        "Info",
        "Debug",
        "Trace",
    };

    Config config;
    Sakura::MenuBar menuBar;
    Sakura::Menu appMenu;
    Sakura::Menu flowgraphMenu;
    Sakura::Menu viewMenu;
    Sakura::Menu themeMenu;
    Sakura::Menu developerMenu;
    Sakura::Menu logLevelMenu;
    Sakura::Menu helpMenu;
    Sakura::MenuItem aboutItem;
    Sakura::MenuItem licenseItem;
    Sakura::MenuItem thirdPartyItem;
    Sakura::MenuItem checkUpdatesItem;
    Sakura::MenuItem preferencesItem;
    Sakura::MenuItem quitItem;
    Sakura::MenuItem newFlowgraphItem;
    Sakura::MenuItem openFlowgraphItem;
    Sakura::MenuItem saveFlowgraphItem;
    Sakura::MenuItem flowgraphInfoItem;
    Sakura::MenuItem closeFlowgraphItem;
    Sakura::MenuItem renameFlowgraphItem;
    Sakura::MenuItem openExamplesItem;
    Sakura::MenuItem infoPanelItem;
    Sakura::MenuItem backgroundParticlesItem;
    std::vector<Sakura::MenuItem> themeItems;
    Sakura::MenuItem remoteStreamingItem;
    Sakura::MenuItem openSettingsItem;
    Sakura::MenuItem latencyWindowItem;
    Sakura::MenuItem runtimeMetricsItem;
    Sakura::MenuItem benchmarksItem;
    std::vector<Sakura::MenuItem> logLevelItems;
    Sakura::MenuItem openDeveloperSettingsItem;
    Sakura::MenuItem gettingStartedItem;
    Sakura::MenuItem twitterItem;
    Sakura::MenuItem documentationItem;
    Sakura::MenuItem repositoryItem;
    Sakura::MenuItem reportIssueItem;
    std::array<Sakura::Divider, 9> dividers;
    Sakura::KeyboardInput shortcuts;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MENUBAR_HH
