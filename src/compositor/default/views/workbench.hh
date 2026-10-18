#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_WORKBENCH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_WORKBENCH_HH

#include "flowgraph/key_value.hh"
#include "flowgraph/pane.hh"
#include "file_picker.hh"
#include "hud/info.hh"
#include "hud/welcome.hh"
#include "modal/container.hh"
#include "start_tab.hh"
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
                     std::optional<std::string> activate = std::nullopt) {
        workspace.update({
            .tabs = true,
            .accepts = "main",
            .panes = std::move(panes),
            .activate = std::move(activate),
            .leading = &titlebar.leading(),
            .trailing = &titlebar.trailing(),
            .end = &titlebar.end(),
            .tabBarHeight = titlebar.height(),
            .leadingInset = titlebar.leadingInset(),
            .onTabMenu = [this](const std::string& pane, Extent2D<F32> point) {
                if (pane != StartTabView::PaneId) {
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
        std::optional<InfoHudView::Config> infoHud;
        std::optional<WelcomeHudView::Config> welcomeHud;
        std::optional<StartTabView::Config> startTab;
        U64 startTabRequests = 0;
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

        std::vector<Sakura::HudLayer::Hud> hudItems;
        if (this->config.welcomeHud.has_value()) {
            welcomeHud.update(std::move(this->config.welcomeHud.value()));
            hudItems.push_back(welcomeHud.hud());
        }
        if (this->config.infoHud.has_value()) {
            infoHud.update(std::move(this->config.infoHud.value()));
            hudItems.push_back(infoHud.hud());
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
            if (this->config.startTabRequests != startTabRequests) {
                activate = StartTabView::PaneId;
            }
        }
        startTabRequests = this->config.startTabRequests;
        workspaceShell.updatePanes(std::move(panes), std::move(activate));
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

        root.update({
            .id = "workbench",
            .layers = {&background, &content, &floating, &huds, &toasts, &modalLayer, &filePickerLayer},
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
    InfoHudView infoHud;
    WelcomeHudView welcomeHud;
    StartTabView startTab;
    U64 startTabRequests = 0;
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
