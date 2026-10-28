#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_TITLEBAR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_TITLEBAR_HH

#include "components/assistant_composer.hh"

#include "jetstream/render/sakura/base.hh"

#include "render/sakura/titlebar.hh"

#include <algorithm>
#include <array>
#include <functional>
#include <optional>
#include <string>
#include <unordered_set>
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
        ToggleAgent,
        ToggleVoice,
        GettingStarted,
        Documentation,
        OpenRepository,
        ReportIssue,
        OpenTwitter,
        SendFeedback,
        Quit,
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
        ToggleEnvironment,
        Close,
    };

    static constexpr F32 IndicatorGap = 8.0f;
    static constexpr F32 NewTabRounding = 14.0f;

    struct VoiceIndicator {
        using Phase = Sakura::Voice::Phase;

        Phase phase = Phase::Idle;
        F32 level = 0.0f;
        std::function<void()> onStart;
        std::function<void()> onStop;
    };

    struct Config {
        F32 leadingInset = 0.0f;
        F32 height = 0.0f;
        bool remoteStarted = false;
        bool updateAvailable = false;
        bool updateReady = false;
        bool intelligenceEnabled = false;
        bool agentUnread = false;
        std::optional<VoiceIndicator> voice;
        std::unordered_set<std::string> environmentFlowgraphs;
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
            std::move(updateEntry),
            separator,
            {.label = "Settings", .shortcut = "CTRL+,", .onClick = action(Action::Settings)},
            separator,
        };
        appEntries.push_back({
            .label = "Help",
            .children = {
                {.label = "Getting Started", .onClick = action(Action::GettingStarted)},
                {.label = "Documentation", .onClick = action(Action::Documentation)},
                {.label = "Repository", .onClick = action(Action::OpenRepository)},
                {.label = "Report Issue", .onClick = action(Action::ReportIssue)},
                {.label = "Luigi's Twitter", .onClick = action(Action::OpenTwitter)},
            },
        });
        appEntries.push_back({.label = "Send Feedback", .shortcut = "CTRL+G", .onClick = action(Action::SendFeedback)});
        appEntries.push_back(separator);
        appEntries.push_back({
            .label = "Developer",
            .children = developerEntries(action, separator),
        });
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
            .icon = ICON_LC_RADIO_TOWER,
            .label = "Remote Sharing",
            .colorKey = "action_btn_text",
            .backgroundColorKey = "remote_btn",
            .hoverColorKey = "remote_btn_hovered",
            .raised = true,
            .onClick = action(Action::RemoteStreaming),
        });
        updateButton.update({
            .icon = ICON_LC_CIRCLE_ARROW_UP,
            .label = this->config.updateReady ? "Update Ready" : "Update Available",
            .colorKey = "action_btn_text",
            .backgroundColorKey = "action_btn",
            .hoverColorKey = "action_btn_hovered",
            .raised = true,
            .onClick = updatePending ? action(Action::OpenUpdateModal) : std::function<void()>(),
        });
        helpButton.update({
            .icon = ICON_LC_CIRCLE_QUESTION_MARK,
            .colorKey = "text_secondary",
            .backgroundColorKey = "transparent",
            .hoverColorKey = "chrome_button_hover",
            .onClick = action(Action::Documentation),
        });
        assistantButton.update({
            .icon = ICON_LC_WAND_SPARKLES,
            .colorKey = this->config.agentUnread ? "cyber_blue" : "text_secondary",
            .backgroundColorKey = "transparent",
            .hoverColorKey = "chrome_button_hover",
            .onClick = action(Action::ToggleAgent),
        });
        std::vector<Sakura::Stack::Item> indicators;
        if (this->config.remoteStarted) {
            indicators.push_back({.child = &remoteButton});
        }
        if (updatePending) {
            indicators.push_back({.child = &updateButton});
        }
#ifndef JST_OS_BROWSER
        if (this->config.voice.has_value()) {
            const auto& voice = *this->config.voice;
            voicePill.update({
                .phase = voice.phase,
                .level = voice.level,
                .startHint = "Click to reconnect",
                .stopHint = "Click to end",
                .onStart = voice.onStart,
                .onStop = voice.onStop,
            });
            indicators.push_back({.child = &voicePill});
        } else if (this->config.intelligenceEnabled) {
            indicators.push_back({.child = &assistantButton});
        }
#endif
        indicators.push_back({.child = &helpButton});
        indicatorRow.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = IndicatorGap,
            .items = std::move(indicators),
        });

        plus.update({
            .colorKey = "pill_text",
            .glyph = Sakura::ChromeButton::Glyph::Plus,
            .rounding = NewTabRounding,
            .onClick = action(Action::NewTab),
        });

        updateTabMenu();
    }

    std::vector<Sakura::Shortcuts::Binding> shortcuts() {
        std::vector<Sakura::Shortcuts::Binding> bindings = {
            bind(KeyCode::T, Action::NewTab),
            bind(KeyCode::O, Action::OpenFlowgraph),
            bind(KeyCode::S, Action::SaveFlowgraph),
            bind(KeyCode::W, Action::CloseFlowgraph),
            bind(KeyCode::I, Action::ShowFlowgraphInfo),
            bind(KeyCode::G, Action::SendFeedback),
            bind(KeyCode::Comma, Action::Settings),
        };
#ifndef JST_OS_BROWSER
        if (config.intelligenceEnabled) {
            bindings.push_back(bind(AssistantComposer::VoiceKey, Action::ToggleVoice));
        }
#endif
        return bindings;
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
                    .label = "Environment View",
                    .selected = config.environmentFlowgraphs.contains(tabFlowgraph),
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
    Sakura::PillButton assistantButton;
    Sakura::Voice voicePill;
    Sakura::PillButton helpButton;
    Sakura::Stack indicatorRow;
    Sakura::Menu tabMenu;
    std::string tabFlowgraph;
    Extent2D<F32> tabPoint = {0.0f, 0.0f};
    bool tabOpen = false;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_TITLEBAR_HH
