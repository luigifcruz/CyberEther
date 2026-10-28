#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_WORKBENCH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_WORKBENCH_HH

#include "agent/chat.hh"
#include "agent/voice.hh"
#include "flowgraph/key_value.hh"
#include "flowgraph/pane.hh"
#include "file_picker.hh"
#include "modal/container.hh"
#include "docs.hh"
#include "settings/base.hh"
#include "start_tab/base.hh"
#include "titlebar.hh"

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Jetstream {

struct WorkspaceShellView {
    void updateTitlebar(TitlebarView::Config config) {
        titlebar.update(std::move(config));
    }

    void updatePanes(std::vector<Sakura::Dock::Pane> panes,
                     std::optional<std::string> activate = std::nullopt,
                     bool hideTabs = false) {
        workspace.update({
            .tabs = true,
            .hideTabs = hideTabs,
            .accepts = "main",
            .panes = std::move(panes),
            .activate = std::move(activate),
            .leading = &titlebar.leading(),
            .trailing = &titlebar.trailing(),
            .end = &titlebar.end(),
            .tabBarHeight = titlebar.height(),
            .leadingInset = titlebar.leadingInset(),
            .onTabMenu = [this](const std::string& pane, Extent2D<F32> point) {
                if (pane != StartTabView::PaneId && pane != SettingsView::PaneId && pane != DocsView::PaneId) {
                    titlebar.openTabMenu(pane, point);
                }
            },
            .onBarPress = [this](U8 clicks) {
                titlebar.barPress(clicks);
            },
        });
        shell.update({
            .items = {
                {.child = &workspace},
                {.child = &titlebar.tabMenuComponent(), .anchor = Sakura::Anchor::TopLeft},
            },
        });
        keys.update({
            .bindings = titlebar.shortcuts(),
            .content = &shell,
        });
    }

    Sakura::Component& component() {
        return keys;
    }

    TitlebarView& bar() {
        return titlebar;
    }

    const TitlebarView& bar() const {
        return titlebar;
    }

    Sakura::Dock& dock() {
        return workspace;
    }

 private:
    TitlebarView titlebar;
    Sakura::Dock workspace;
    Sakura::Overlay shell;
    Sakura::Shortcuts keys;
};

struct WorkbenchView {
    struct Config {
        bool filePending = false;
        bool backgroundParticles = true;
        AgentChatView::Config agentChat;
        AgentVoiceView::Config agentVoice;
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
        ModalView::Config modal;
        std::optional<FlowgraphKeyValueWindow::Config> flowgraphMetadata;
        std::optional<FlowgraphKeyValueWindow::Config> flowgraphEnvironment;
        std::vector<FlowgraphPane::Config> flowgraphs;
        FilePickerView::Config filePicker;
    };

    void update(Config config) {
        this->config = std::move(config);
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

        this->config.agentChat.topOffset = workspaceShell.bar().height();
        agentChat.update(std::move(this->config.agentChat));
        agentVoice.update(std::move(this->config.agentVoice));

        std::vector<Sakura::HudLayer::Hud> hudItems;
        if (agentVoice.shown()) {
            hudItems.push_back(agentVoice.hud());
        }
        huds.update({.huds = std::move(hudItems)});

        flowgraphOrder.clear();
        std::unordered_set<std::string> activeFlowgraphs;
        for (auto& flowgraphConfig : this->config.flowgraphs) {
            flowgraphOrder.push_back(flowgraphConfig.id);
            activeFlowgraphs.insert(flowgraphConfig.id);
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
        workspaceShell.updatePanes(std::move(panes), std::move(activate), this->config.hideTabs);
        content.update({.content = &workspaceShell.component()});

        if (this->config.flowgraphMetadata.has_value()) {
            flowgraphMetadataWindow.update(std::move(this->config.flowgraphMetadata.value()));
            windows.push_back(flowgraphMetadataWindow.window());
        }
        if (this->config.flowgraphEnvironment.has_value()) {
            flowgraphEnvironmentWindow.update(std::move(this->config.flowgraphEnvironment.value()));
            windows.push_back(flowgraphEnvironmentWindow.window());
        }
        floating.update({.windows = std::move(windows)});

        toasts.update();

        modal.update(std::move(this->config.modal));
        modalLayer.update(modal.layer());
        filePicker.update(std::move(this->config.filePicker));
        filePickerLayer.update(filePicker.layer());

        std::vector<Sakura::Layer*> layers = {&background, &content, &floating, &huds};
        if (agentChat.shown() && !agentVoice.shown()) {
            layers.push_back(&agentChat.layer());
        }
        layers.push_back(&toasts);
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
    Sakura::ModalLayer modalLayer;
    Sakura::ModalLayer filePickerLayer;
    Sakura::Text filePendingText;
    Sakura::Panel filePendingPanel;
    AgentChatView agentChat;
    AgentVoiceView agentVoice;
    StartTabView startTab;
    U64 startTabRequests = 0;
    SettingsView settings;
    U64 settingsTabRequests = 0;
    DocsView docs;
    U64 docsTabRequests = 0;
    WorkspaceShellView workspaceShell;
    std::vector<std::string> flowgraphOrder;
    std::unordered_map<std::string, FlowgraphPane> flowgraphPanes;
    FlowgraphKeyValueWindow flowgraphMetadataWindow;
    FlowgraphKeyValueWindow flowgraphEnvironmentWindow;
    ModalView modal;
    FilePickerView filePicker;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_WORKBENCH_HH
