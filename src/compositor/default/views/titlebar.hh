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

    void update(Config config);
    std::vector<Sakura::Shortcuts::Binding> shortcuts();
    void openTabMenu(const std::string& flowgraph, Extent2D<F32> point);
    void barPress(U8 clicks) const;
    Sakura::Component& leading();
    F32 leadingReach() const;
    Sakura::Component& trailing();
    Sakura::Component& end();
    Sakura::Component& tabMenuComponent();
    F32 height() const;
    F32 leadingInset() const;
    F32 trailingInset() const;

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

    Sakura::Shortcuts::Binding bind(KeyCode key, Action value);
    Sakura::Shortcuts::Binding bindEdit(KeyCode key, Sakura::Shortcuts::Modifier modifier, Action value);
    void updateTabMenu();
    void emit(Action value) const;

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
