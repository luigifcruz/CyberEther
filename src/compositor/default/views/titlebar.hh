#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_TITLEBAR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_TITLEBAR_HH

#include "jetstream/render/sakura/base.hh"

#include "render/sakura/titlebar.hh"

#include <algorithm>
#include <array>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct TitlebarView {
    static constexpr F32 MinimumHeight = Sakura::Titlebar::MinimumHeight;

    enum class Action {
        About,
        Settings,
        CheckForUpdates,
        OpenUpdateModal,
        RemoteStreaming,
        GettingStarted,
        Documentation,
        OpenRepository,
        ReportIssue,
        OpenTwitter,
        SendFeedback,
        Quit,
        NewFlowgraph,
        OpenFlowgraph,
        SaveFlowgraph,
        ShowFlowgraphInfo,
        CloseFlowgraph,
        ToggleTiming,
        ShowBenchmarks,
        ReloadPlugins,
        OpenDeveloperSettings,
        NewTab,
    };

    enum class TabAction {
        Save,
        Rename,
        Info,
        ToggleMetadata,
        ToggleEnvironment,
        Close,
    };

    static constexpr F32 IndicatorGap = 8.0f;

    struct Config {
        std::string id;
        F32 leadingInset = 0.0f;
        F32 height = 0.0f;
        bool remoteSupported = false;
        bool remoteStarted = false;
        bool updateAvailable = false;
        bool updateReady = false;
        std::optional<std::string> focusedFlowgraph;
        bool flowgraphMetadataVisible = false;
        bool flowgraphEnvironmentVisible = false;
        bool debugTimingEnabled = false;
        I32 debugLogLevel = 0;
        std::function<void(Action)> onAction;
        std::function<void(I32)> onLogLevel;
        std::function<void(TabAction, const std::string&)> onTabAction;
        std::function<void(U8)> onBarPress;
    };

    void update(Config config) {
        this->config = std::move(config);

        using Entry = Sakura::MenuEntry;
        const auto action = [this](Action value) {
            return [this, value]() { emit(value); };
        };
        const Entry separator = {.separator = true};

        Entry updateEntry = {.label = "Check for Updates", .onClick = action(Action::CheckForUpdates)};
        if (this->config.updateReady || this->config.updateAvailable) {
            updateEntry = {
                .label = this->config.updateReady ? "Update Ready" : "Update Available",
                .colorKey = "warning_yellow",
                .onClick = action(Action::OpenUpdateModal),
            };
        }
        std::vector<Entry> appEntries = {
            {.label = "About CyberEther", .onClick = action(Action::About)},
            {.label = "Settings", .shortcut = "CTRL+,", .onClick = action(Action::Settings)},
            std::move(updateEntry),
            {
                .label = "Remote Streaming",
                .enabled = this->config.remoteSupported,
                .onClick = action(Action::RemoteStreaming),
            },
            {
                .label = "Developer",
                .children = developerEntries(action, separator),
            },
            {
                .label = "Help",
                .children = {
                    {.label = "Getting Started", .onClick = action(Action::GettingStarted)},
                    {.label = "Documentation", .onClick = action(Action::Documentation)},
                    {.label = "Repository", .onClick = action(Action::OpenRepository)},
                    {.label = "Report Issue", .onClick = action(Action::ReportIssue)},
                    {.label = "Luigi's Twitter", .onClick = action(Action::OpenTwitter)},
                },
            },
            {.label = "Send Feedback", .shortcut = "CTRL+G", .onClick = action(Action::SendFeedback)},
        };
#ifndef JST_OS_BROWSER
        appEntries.push_back(separator);
        appEntries.push_back({.label = "Quit CyberEther", .onClick = action(Action::Quit)});
#endif
        wordmark.update({
            .label = "CyberEther",
            .font = Sakura::Text::Font::Display,
            .scale = 1.6f,
            .colorKey = "cyber_blue",
            .entries = std::move(appEntries),
        });

        const bool updatePending = this->config.updateReady || this->config.updateAvailable;
        remoteButton.update({
            .icon = ICON_FA_TOWER_BROADCAST,
            .label = "Remote Sharing",
            .colorKey = "action_btn_text",
            .backgroundColorKey = "remote_btn",
            .hoverColorKey = "remote_btn_hovered",
            .raised = true,
            .onClick = action(Action::RemoteStreaming),
        });
        updateButton.update({
            .icon = ICON_FA_CIRCLE_ARROW_UP,
            .label = this->config.updateReady ? "Update Ready" : "Update Available",
            .colorKey = "action_btn_text",
            .backgroundColorKey = "action_btn",
            .hoverColorKey = "action_btn_hovered",
            .raised = true,
            .onClick = updatePending ? action(Action::OpenUpdateModal) : std::function<void()>(),
        });
        helpButton.update({
            .icon = ICON_FA_CIRCLE_QUESTION,
            .colorKey = "text_secondary",
            .backgroundColorKey = "transparent",
            .hoverColorKey = "chrome_button_hover",
            .onClick = action(Action::SendFeedback),
        });
        std::vector<Sakura::Stack::Item> indicators;
        if (this->config.remoteStarted) {
            indicators.push_back({.child = &remoteButton});
        }
        if (updatePending) {
            indicators.push_back({.child = &updateButton});
        }
        indicators.push_back({.child = &helpButton});
        indicatorRow.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = IndicatorGap,
            .items = std::move(indicators),
        });

        plus.update({
            .colorKey = "pill_text",
            .glyph = Sakura::ChromeButton::Glyph::Plus,
            .onClick = action(Action::NewTab),
        });

        updateTabMenu();
    }

    std::vector<Sakura::Shortcuts::Binding> shortcuts() {
        return {
            bind(KeyCode::N, Action::NewFlowgraph),
            bind(KeyCode::O, Action::OpenFlowgraph),
            bind(KeyCode::S, Action::SaveFlowgraph),
            bind(KeyCode::W, Action::CloseFlowgraph),
            bind(KeyCode::I, Action::ShowFlowgraphInfo),
            bind(KeyCode::G, Action::SendFeedback),
            bind(KeyCode::Comma, Action::Settings),
        };
    }

    void openTabMenu(const std::string& flowgraph, Extent2D<F32> point) {
        tabFlowgraph = flowgraph;
        tabPoint = point;
        tabOpen = true;
        updateTabMenu();
    }

    void barPress(U8 clicks) const {
        if (config.onBarPress) {
            config.onBarPress(clicks);
        }
    }

    Sakura::Component& leading() {
        return wordmark;
    }

    Sakura::Component& trailing() {
        return plus;
    }

    Sakura::Component& end() {
        return indicatorRow;
    }

    Sakura::Component& tabMenuComponent() {
        return tabMenu;
    }

    F32 height() const {
        return config.height > 0.0f ? config.height : MinimumHeight;
    }

    F32 leadingInset() const {
        return config.leadingInset;
    }

 private:
    template<typename ActionFor>
    std::vector<Sakura::MenuEntry> developerEntries(const ActionFor& action,
                                                    const Sakura::MenuEntry& separator) const {
        using Entry = Sakura::MenuEntry;
        std::vector<Entry> levels;
        for (U64 index = 0; index < LogLevels.size(); ++index) {
            levels.push_back({
                .label = LogLevels[index],
                .selected = config.debugLogLevel == static_cast<I32>(index),
                .onClick = [onLogLevel = config.onLogLevel, index]() {
                    if (onLogLevel) {
                        onLogLevel(static_cast<I32>(index));
                    }
                },
            });
        }
        return {
            {.label = "Show Timing", .selected = config.debugTimingEnabled, .onClick = action(Action::ToggleTiming)},
            {.label = "Show Benchmarks", .onClick = action(Action::ShowBenchmarks)},
            separator,
            {.label = "Reload Plugins", .onClick = action(Action::ReloadPlugins)},
            separator,
            {.label = "Log Level", .children = std::move(levels)},
            separator,
            {.label = "Open In Settings", .onClick = action(Action::OpenDeveloperSettings)},
        };
    }

    static constexpr std::array<const char*, 5> LogLevels = {
        "Fatal / Error",
        "Warnings",
        "Info",
        "Debug",
        "Trace",
    };

    Sakura::Shortcuts::Binding bind(KeyCode key, Action value) {
        return {
            .key = key,
            .modifier = Sakura::Shortcuts::Modifier::Primary,
            .onPressed = [this, value]() { emit(value); },
        };
    }

    void updateTabMenu() {
        using Entry = Sakura::MenuEntry;
        const bool focused = config.focusedFlowgraph == tabFlowgraph;
        const auto tab = [this](TabAction value) {
            return [this, value]() {
                if (config.onTabAction && !tabFlowgraph.empty()) {
                    config.onTabAction(value, tabFlowgraph);
                }
            };
        };
        tabMenu.update({
            .entries = {
                {.label = "Save", .onClick = tab(TabAction::Save)},
                {.label = "Rename", .onClick = tab(TabAction::Rename)},
                {.label = "Info", .onClick = tab(TabAction::Info)},
                {
                    .label = "Metadata View",
                    .selected = focused && config.flowgraphMetadataVisible,
                    .onClick = tab(TabAction::ToggleMetadata),
                },
                {
                    .label = "Environment View",
                    .selected = focused && config.flowgraphEnvironmentVisible,
                    .onClick = tab(TabAction::ToggleEnvironment),
                },
                Entry{.separator = true},
                {.label = "Close", .onClick = tab(TabAction::Close)},
            },
            .open = tabOpen,
            .atCursor = false,
            .point = tabPoint,
            .onClose = [this]() {
                tabOpen = false;
            },
        });
    }

    void emit(Action value) const {
        if (config.onAction) {
            config.onAction(value);
        }
    }

    Config config;
    Sakura::ChromeButton wordmark;
    Sakura::ChromeButton plus;
    Sakura::PillButton remoteButton;
    Sakura::PillButton updateButton;
    Sakura::PillButton helpButton;
    Sakura::Stack indicatorRow;
    Sakura::Menu tabMenu;
    std::string tabFlowgraph;
    Extent2D<F32> tabPoint = {0.0f, 0.0f};
    bool tabOpen = false;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_TITLEBAR_HH
