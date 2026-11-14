#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_TITLEBAR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_TITLEBAR_HH

#include "components/assistant_composer.hh"
#include "components/sidebar_tiles.hh"

#include "jetstream/render/sakura/base.hh"

#include "render/sakura/titlebar.hh"

#include <algorithm>
#include <array>
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
        WhatsNew,
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
        SaveFlowgraphAs,
        ShowFlowgraphInfo,
        CloseFlowgraph,
        ToggleTiming,
        ShowBenchmarks,
        ReloadPlugins,
        OpenDeveloperSettings,
        NewTab,
        OpenTerminal,
        ToggleSidebar,
        Undo,
        Redo,
        TogglePause,
    };

    enum class TabAction {
        Save,
        SaveAs,
        Rename,
        Info,
        ToggleEnvironment,
        Close,
    };

    static constexpr F32 IndicatorGap = Sakura::Spacing::Small;
    static constexpr F32 TrailingInset = 2.0f;
    static constexpr F32 SidebarButtonSize = 24.0f;
    static constexpr F32 SidebarIconSize = 14.0f;
    static constexpr F32 SidebarDrop = 1.0f;
    static constexpr F32 NewTabRounding = 14.0f;
    static constexpr F32 WordmarkEmboss = 0.0125f;

    struct VoiceIndicator {
        using Phase = Sakura::Voice::Phase;

        Phase phase = Phase::Idle;
        F32 level = 0.0f;
        bool muted = false;
        Sakura::Callback<void()> onStart;
        Sakura::Callback<void()> onStop;
        Sakura::Callback<void()> onMute;
    };

    struct Config {
        F32 leadingInset = 0.0f;
        F32 height = 0.0f;
        F32 trailingInset = 0.0f;
        bool remoteStarted = false;
        bool updateAvailable = false;
        bool updateReady = false;
        bool intelligenceEnabled = false;
        bool agentWorking = false;
        bool agentUnread = false;
        bool terminalEnabled = false;
        std::optional<VoiceIndicator> voice;
        std::unordered_set<std::string> environmentFlowgraphs;
        bool debugTimingEnabled = false;
        I32 debugLogLevel = 0;
        std::optional<bool> sidebarOpen;
        Sakura::Callback<void(Action)> onAction;
        Sakura::Callback<void(I32)> onLogLevel;
        Sakura::Callback<void(TabAction, const std::string&)> onTabAction;
        Sakura::Callback<void(U8)> onBarPress;
    };

    void update(Config config) {
        this->config = std::move(config);

        using Entry = Sakura::MenuEntry;
        using Modifier = Sakura::Shortcuts::Modifier;
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
            {.label = "What's New", .onClick = action(Action::WhatsNew)},
            std::move(updateEntry),
            separator,
            {
                .label = "Settings",
                .shortcut = Sakura::Shortcuts::Label(Modifier::Primary, ","),
                .onClick = action(Action::Settings),
            },
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
        appEntries.push_back({
            .label = "Send Feedback",
            .shortcut = Sakura::Shortcuts::Label(Modifier::Primary, "G"),
            .onClick = action(Action::SendFeedback),
        });
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
            .scale = Sakura::Typography::Scale::Headline,
            .emboss = WordmarkEmboss,
            .colorKey = "cyber_blue",
            .entries = std::move(appEntries),
        });

        const bool updatePending = this->config.updateReady || this->config.updateAvailable;
        remoteButton.update({
            .icon = ICON_LC_SCREEN_SHARE,
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
            .onClick = updatePending ? action(Action::OpenUpdateModal) : Sakura::Callback<void()>(),
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
            .colorKey = "text_secondary",
            .backgroundColorKey = "transparent",
            .hoverColorKey = "chrome_button_hover",
            .onClick = action(Action::ToggleAgent),
        });
        assistantPill.update({
            .state = this->config.agentWorking ? Sakura::AssistantPill::State::Working
                                               : Sakura::AssistantPill::State::Unread,
            .icon = ICON_LC_WAND_SPARKLES,
            .label = this->config.agentWorking ? "Working" : "New reply",
            .onClick = action(Action::ToggleAgent),
        });
        terminalButton.update({
            .icon = ICON_LC_SQUARE_TERMINAL,
            .colorKey = "text_secondary",
            .backgroundColorKey = "transparent",
            .hoverColorKey = "chrome_button_hover",
            .onClick = action(Action::OpenTerminal),
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
                .muted = voice.muted,
                .startHint = "Click to reconnect",
                .stopHint = "Click to end",
                .onStart = voice.onStart,
                .onStop = voice.onStop,
                .onMute = voice.onMute,
            });
            indicators.push_back({.child = &voicePill});
        } else if (this->config.intelligenceEnabled) {
            if (this->config.agentWorking || this->config.agentUnread) {
                indicators.push_back({.child = &assistantPill});
            } else {
                indicators.push_back({.child = &assistantButton});
            }
        }
#endif
        if (this->config.terminalEnabled) {
            indicators.push_back({.child = &terminalButton});
        }
        indicators.push_back({.child = &helpButton});
        indicatorRow.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = IndicatorGap,
            .items = std::move(indicators),
        });

        sidebarIcon.update({
            .str = this->config.sidebarOpen.value_or(true) ? ICON_LC_PANEL_LEFT_CLOSE : ICON_LC_PANEL_LEFT_OPEN,
            .tone = Sakura::Text::Tone::Secondary,
            .scale = SidebarIconSize / Sakura::Typography::FontSize,
        });
        sidebarButton.update({
            .content = &sidebarIcon,
            .size = {SidebarButtonSize, SidebarButtonSize},
            .padding = (SidebarButtonSize - SidebarIconSize) * 0.5f,
            .rounding = SidebarButtonSize * 0.5f,
            .border = false,
            .colorKey = "transparent",
            .hoveredColorKey = "chrome_button_hover",
            .onClick = action(Action::ToggleSidebar),
        });
        sidebarDrop.update({.size = {0.0f, 0.0f}, .border = false, .colorKey = "transparent"});
        sidebarSlot.update({
            .spacing = 0.0f,
            .items = {
                {.child = &sidebarDrop, .size = 2.0f * SidebarDrop},
                {.child = &sidebarButton},
            },
        });
        sidebarGap.update({.size = {0.0f, 0.0f}, .border = false, .colorKey = "transparent"});
        std::vector<Sakura::Stack::Item> leadingItems = {{.child = &wordmark}};
        if (this->config.sidebarOpen.has_value()) {
            leadingItems.push_back({.child = &sidebarGap, .grow = 1.0f});
            leadingItems.push_back({.child = &sidebarSlot});
        }
        leadingRow.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = 0.0f,
            .items = std::move(leadingItems),
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
            bindEdit(KeyCode::Z, Sakura::Shortcuts::Modifier::Primary, Action::Undo),
            bindEdit(KeyCode::Z, Sakura::Shortcuts::Modifier::PrimaryShift, Action::Redo),
            bindEdit(KeyCode::Y, Sakura::Shortcuts::Modifier::Primary, Action::Redo),
            {
                .key = KeyCode::Space,
                .modifier = Sakura::Shortcuts::Modifier::None,
                .yieldsToText = true,
                .onPressed = [this]() { emit(Action::TogglePause); },
            },
            bind(KeyCode::T, Action::NewTab),
            bind(KeyCode::O, Action::OpenFlowgraph),
            bind(KeyCode::S, Action::SaveFlowgraph),
            {
                .key = KeyCode::S,
                .modifier = Sakura::Shortcuts::Modifier::PrimaryShift,
                .onPressed = [this]() { emit(Action::SaveFlowgraphAs); },
            },
            bind(KeyCode::W, Action::CloseFlowgraph),
            bind(KeyCode::I, Action::ShowFlowgraphInfo),
            bind(KeyCode::G, Action::SendFeedback),
            bind(KeyCode::Comma, Action::Settings),
            bind(KeyCode::GraveAccent, Action::OpenTerminal),
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
        return leadingRow;
    }

    F32 leadingReach() const {
        return config.sidebarOpen.value_or(false) ? Sakura::Dock::ContentPadding + SidebarTiles::SidebarWidth : 0.0f;
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

    F32 trailingInset() const {
        return std::max(config.trailingInset, TrailingInset);
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

    Sakura::Shortcuts::Binding bindEdit(KeyCode key, Sakura::Shortcuts::Modifier modifier, Action value) {
        return {
            .key = key,
            .modifier = modifier,
            .repeat = true,
            .yieldsToText = true,
            .onPressed = [this, value]() { emit(value); },
        };
    }

    void updateTabMenu() {
        using Entry = Sakura::MenuEntry;
        using Modifier = Sakura::Shortcuts::Modifier;
        const auto tab = [this](TabAction value) {
            return [this, value]() {
                if (config.onTabAction && !tabFlowgraph.empty()) {
                    config.onTabAction(value, tabFlowgraph);
                }
            };
        };
        tabMenu.update({
            .entries = {
                {
                    .label = "Save",
                    .shortcut = Sakura::Shortcuts::Label(Modifier::Primary, "S"),
                    .onClick = tab(TabAction::Save),
                },
                {
                    .label = "Save As",
                    .shortcut = Sakura::Shortcuts::Label(Modifier::PrimaryShift, "S"),
                    .onClick = tab(TabAction::SaveAs),
                },
                {.label = "Rename", .onClick = tab(TabAction::Rename)},
                {
                    .label = "Info",
                    .shortcut = Sakura::Shortcuts::Label(Modifier::Primary, "I"),
                    .onClick = tab(TabAction::Info),
                },
                {
                    .label = "Environment View",
                    .selected = config.environmentFlowgraphs.contains(tabFlowgraph),
                    .onClick = tab(TabAction::ToggleEnvironment),
                },
                Entry{.separator = true},
                {
                    .label = "Close",
                    .shortcut = Sakura::Shortcuts::Label(Modifier::Primary, "W"),
                    .onClick = tab(TabAction::Close),
                },
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
    Sakura::Div sidebarGap;
    Sakura::Text sidebarIcon;
    Sakura::Div sidebarButton;
    Sakura::Div sidebarDrop;
    Sakura::Stack sidebarSlot;
    Sakura::Stack leadingRow;
    Sakura::ChromeButton plus;
    Sakura::PillButton remoteButton;
    Sakura::PillButton updateButton;
    Sakura::PillButton assistantButton;
    Sakura::AssistantPill assistantPill;
    Sakura::Voice voicePill;
    Sakura::PillButton terminalButton;
    Sakura::PillButton helpButton;
    Sakura::Stack indicatorRow;
    Sakura::Menu tabMenu;
    std::string tabFlowgraph;
    Extent2D<F32> tabPoint = {0.0f, 0.0f};
    bool tabOpen = false;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_TITLEBAR_HH
