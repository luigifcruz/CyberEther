#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_WORKBENCH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_WORKBENCH_HH

#include "agent/chat.hh"
#include "flowgraph/close_prompt.hh"
#include "flowgraph/pane.hh"
#include "file_picker.hh"
#include "modal/container.hh"
#include "docs.hh"
#include "settings/base.hh"
#include "start_tab/base.hh"
#include "terminal.hh"
#include "titlebar.hh"

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Jetstream {

struct WorkspaceShellView {
    static constexpr F32 PromptGap = Sakura::Spacing::Small;

    struct Config {
        std::vector<Sakura::Dock::Pane> panes;
        std::optional<std::string> activate;
        bool hideTabs = false;
        Sakura::Component* aside = nullptr;
        std::optional<FlowgraphClosePromptView::Config> closePrompt;
    };

    void updateTitlebar(TitlebarView::Config config);
    void updatePanes(Config config);
    Sakura::Component& component();
    Sakura::ModalLayer::Config promptLayer();
    TitlebarView& bar();
    Sakura::Dock& dock();

 private:
    TitlebarView titlebar;
    Sakura::Dock workspace;
    FlowgraphClosePromptView closing;
    std::optional<Rect> promptAnchor;
    Sakura::Overlay shell;
    Sakura::Shortcuts keys;
};

struct WorkbenchView {
    struct Config {
        bool filePending = false;
        bool backgroundParticles = true;
        AgentChatView::Config agentChat;
        std::optional<StartTabView::Config> startTab;
        bool hideTabs = false;
        U64 startTabRequests = 0;
        bool startTabFocused = false;
        std::optional<SettingsView::Config> settings;
        U64 settingsTabRequests = 0;
        bool settingsTabFocused = false;
        std::optional<DocsView::Config> docs;
        U64 docsTabRequests = 0;
        bool docsTabFocused = false;
        TitlebarView::Config titlebar;
        std::optional<FlowgraphClosePromptView::Config> closePrompt;
        ModalView::Config modal;
        std::vector<FlowgraphPane::Config> flowgraphs;
        std::vector<TerminalView::Config> terminals;
        std::vector<TerminalTabView::Config> terminalTabs;
        std::optional<std::string> terminalRequested;
        U64 terminalRequests = 0;
        bool terminalFocused = false;
        FilePickerView::Config filePicker;
    };

    void update(Config config);
    void render(const Sakura::Context& ctx);
    bool dropFiles(const std::vector<std::string>& paths);

 private:
    Config config;
    Sakura::Root root;
    Sakura::BackgroundLayer background;
    Sakura::Layer content;
    Sakura::FloatingLayer floating;
    Sakura::HudLayer huds;
    Sakura::ToastLayer toasts;
    Sakura::ModalLayer promptLayer;
    Sakura::ModalLayer modalLayer;
    Sakura::ModalLayer filePickerLayer;
    Sakura::Text filePendingText;
    Sakura::Panel filePendingPanel;
    AgentChatView agentChat;
    StartTabView startTab;
    U64 startTabRequests = 0;
    bool startTabShown = false;
    U64 startTabVisits = 0;
    SettingsView settings;
    U64 settingsTabRequests = 0;
    DocsView docs;
    U64 docsTabRequests = 0;
    WorkspaceShellView workspaceShell;
    std::vector<std::string> flowgraphOrder;
    std::unordered_map<std::string, FlowgraphPane> flowgraphPanes;
    std::vector<std::string> terminalOrder;
    std::unordered_map<std::string, TerminalView> terminalViews;
    std::unordered_map<std::string, TerminalTabView> terminalTabViews;
    U64 terminalRequests = 0;
    ModalView modal;
    FilePickerView filePicker;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_WORKBENCH_HH
