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

    void updateTitlebar(TitlebarView::Config config) {
        titlebar.update(std::move(config));
    }

    void updatePanes(Config config) {
        std::unordered_set<std::string> terminals;
        for (const auto& pane : config.panes) {
            if (pane.group == TerminalView::Group) {
                terminals.insert(pane.id);
            }
        }
        workspace.update({
            .tabs = true,
            .hideTabs = config.hideTabs,
            .accepts = "main",
            .adopts = {TerminalView::Group},
            .panes = std::move(config.panes),
            .activate = std::move(config.activate),
            .leading = &titlebar.leading(),
            .trailing = &titlebar.trailing(),
            .end = &titlebar.end(),
            .aside = config.aside,
            .asideWidth = AgentChatView::Width,
            .tabBarHeight = titlebar.height(),
            .leadingInset = titlebar.leadingInset(),
            .onTabMenu = [this, terminals = std::move(terminals)](const std::string& pane, Extent2D<F32> point) {
                if (pane != StartTabView::PaneId && pane != SettingsView::PaneId && pane != DocsView::PaneId &&
                    !terminals.contains(pane)) {
                    titlebar.openTabMenu(pane, point);
                }
            },
            .onBarPress = [this](U8 clicks) {
                titlebar.barPress(clicks);
            },
        });
        promptAnchor.reset();
        if (config.closePrompt.has_value()) {
            if (const auto tab = workspace.tabRect(config.closePrompt->flowgraph)) {
                promptAnchor = Rect{tab->x, tab->y, tab->width, tab->height + PromptGap};
            }
            closing.update(std::move(*config.closePrompt));
        } else {
            closing.close();
        }
        shell.update({.items = {
            {.child = &workspace},
            {.child = &titlebar.tabMenuComponent(), .anchor = Sakura::Anchor::TopLeft},
        }});
        keys.update({
            .bindings = titlebar.shortcuts(),
            .content = &shell,
        });
    }

    Sakura::Component& component() {
        return keys;
    }

    Sakura::ModalLayer::Config promptLayer() {
        return closing.layer(promptAnchor);
    }

    TitlebarView& bar() {
        return titlebar;
    }

    Sakura::Dock& dock() {
        return workspace;
    }

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

    void update(Config config) {
        this->config = std::move(config);
        agentChat.update(std::move(this->config.agentChat));
        workspaceShell.updateTitlebar(std::move(this->config.titlebar));

        background.update({
            .particles = this->config.backgroundParticles,
            .topOffset = this->config.filePending ? 0.0f : workspaceShell.bar().height(),
        });

        if (this->config.filePending) {
            filePendingText.update({.str = "Select a file from the browser dialog."});
            filePendingPanel.update({
                .id = "file-pending",
                .content = &filePendingText,
                .size = {0.0f, 0.0f},
                .transparent = true,
            });
            huds.update({
                .huds = {{
                    .content = &filePendingPanel,
                    .anchor = Sakura::Anchor::Center,
                    .borderSize = 0.0f,
                    .backgroundColorKey = "transparent",
                    .borderColorKey = "transparent",
                }},
            });
            root.update({
                .id = "workbench",
                .layers = {&background, &huds},
            });
            return;
        }

        huds.update({});

        std::unordered_set<std::string> activeTerminals;
        for (const auto& terminalConfig : this->config.terminals) {
            activeTerminals.insert(terminalConfig.id);
        }
        for (auto& [id, view] : terminalViews) {
            if (!activeTerminals.contains(id)) {
                view.retire();
            }
        }
        std::unordered_map<std::string, std::vector<Sakura::Dock::Pane>> tiledTerminals;
        std::unordered_map<std::string, std::vector<Sakura::Dock::Pane>> tabTerminals;
        for (auto& terminalConfig : this->config.terminals) {
            const std::string id = terminalConfig.id;
            auto& view = terminalViews[id];
            view.update(std::move(terminalConfig));
            if (view.flowgraph().has_value()) {
                tiledTerminals[*view.flowgraph()].push_back(view.pane());
            } else if (view.tab().has_value()) {
                tabTerminals[*view.tab()].push_back(view.pane());
            }
        }
        terminalOrder.clear();
        std::unordered_set<std::string> activeTabs;
        for (auto& tabConfig : this->config.terminalTabs) {
            const std::string id = tabConfig.id;
            terminalOrder.push_back(id);
            activeTabs.insert(id);
            terminalTabViews[id].update(std::move(tabConfig), std::move(tabTerminals[id]));
        }
        std::erase_if(terminalTabViews, [&](const auto& entry) {
            return !activeTabs.contains(entry.first);
        });
        std::erase_if(terminalViews, [&](const auto& entry) {
            return !activeTerminals.contains(entry.first);
        });

        flowgraphOrder.clear();
        std::unordered_set<std::string> activeFlowgraphs;
        for (auto& flowgraphConfig : this->config.flowgraphs) {
            flowgraphOrder.push_back(flowgraphConfig.id);
            activeFlowgraphs.insert(flowgraphConfig.id);
            flowgraphConfig.adopts = {TerminalView::Group};
            if (auto tiled = tiledTerminals.find(flowgraphConfig.id); tiled != tiledTerminals.end()) {
                flowgraphConfig.terminals = std::move(tiled->second);
            }
            flowgraphPanes[flowgraphConfig.id].update(std::move(flowgraphConfig));
        }
        std::erase_if(flowgraphPanes, [&](const auto& entry) {
            return !activeFlowgraphs.contains(entry.first);
        });

        std::vector<Sakura::Dock::Pane> panes;
        std::vector<Sakura::FloatingLayer::Window> windows;
        for (const auto& flowgraphId : flowgraphOrder) {
            auto& window = flowgraphPanes.at(flowgraphId);
            window.appendPanes(panes);
            window.appendWindows(windows);
        }
        for (const auto& tabId : terminalOrder) {
            panes.push_back(terminalTabViews.at(tabId).pane());
        }
        std::optional<std::string> activate;
        if (this->config.startTab.has_value()) {
            startTab.update(std::move(this->config.startTab.value()));
            panes.push_back(startTab.pane());
            if (this->config.startTabRequests != startTabRequests && this->config.startTabFocused) {
                activate = StartTabView::PaneId;
            }
        }
        startTabRequests = this->config.startTabRequests;
        if (this->config.settings.has_value()) {
            settings.update(std::move(this->config.settings.value()));
            panes.push_back(settings.pane());
            if (this->config.settingsTabRequests != settingsTabRequests && this->config.settingsTabFocused) {
                activate = SettingsView::PaneId;
            }
        }
        settingsTabRequests = this->config.settingsTabRequests;
        if (this->config.docs.has_value()) {
            docs.update(std::move(this->config.docs.value()));
            panes.push_back(docs.pane());
            if (this->config.docsTabRequests != docsTabRequests && this->config.docsTabFocused) {
                activate = DocsView::PaneId;
            }
        }
        docsTabRequests = this->config.docsTabRequests;
        if (this->config.terminalRequests != terminalRequests && this->config.terminalRequested.has_value() &&
            this->config.terminalFocused) {
            activate = this->config.terminalRequested;
        }
        terminalRequests = this->config.terminalRequests;
        workspaceShell.updatePanes({
            .panes = std::move(panes),
            .activate = std::move(activate),
            .hideTabs = this->config.hideTabs,
            .aside = agentChat.shown() ? &agentChat.component() : nullptr,
            .closePrompt = std::move(this->config.closePrompt),
        });
        content.update({.content = &workspaceShell.component()});
        promptLayer.update(workspaceShell.promptLayer());

        floating.update({.windows = std::move(windows)});

        toasts.update();

        modal.update(std::move(this->config.modal));
        modalLayer.update(modal.layer());
        filePicker.update(std::move(this->config.filePicker));
        filePickerLayer.update(filePicker.layer());

        std::vector<Sakura::Layer*> layers = {&background, &content, &floating, &huds};
        layers.push_back(&toasts);
        layers.push_back(&promptLayer);
        layers.push_back(&modalLayer);
        layers.push_back(&filePickerLayer);
        root.update({
            .id = "workbench",
            .layers = std::move(layers),
        });
    }

    void render(const Sakura::Context& ctx) {
        root.render(ctx);
    }

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
