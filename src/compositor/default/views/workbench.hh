#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_WORKBENCH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_WORKBENCH_HH

#include "flowgraph/window.hh"
#include "hud/info.hh"
#include "hud/remote.hh"
#include "hud/welcome.hh"
#include "menubar.hh"
#include "modal/container.hh"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Jetstream {

struct WorkbenchView : public Sakura::Component {
    struct Config {
        bool filePending = false;
        bool backgroundParticles = true;
        bool debugLatencyVisible = false;
        InfoHudView::Config infoHud;
        WelcomeHudView::Config welcomeHud;
        RemoteHudView::Config remoteHud;
        MenubarView::Config menubar;
        ModalView::Config modal;
        std::vector<FlowgraphWindow::Config> flowgraphs;
    };

    void update(Config config) {
        this->config = std::move(config);

        workspaceBackground.update({
            .id = "workspace-background",
            .particles = this->config.backgroundParticles,
            .topOffset = this->config.filePending ? 0.0f : menuBarHeight,
        });

        if (this->config.filePending) {
            filePendingHud.update({
                .id = "file-pending",
                .anchor = Sakura::Hud::Anchor::Center,
                .backgroundAlpha = 0.0f,
                .borderAlpha = 0.0f,
            });
            filePendingText.update({
                .id = "file-pending-text",
                .str = "Select a file from the browser dialog.",
            });
            return;
        }

        infoHud.update(std::move(this->config.infoHud));
        welcomeHud.update(std::move(this->config.welcomeHud));
        remoteHud.update(std::move(this->config.remoteHud));
        notifications.update({.id = "notifications"});
        debugWindow.update({
            .id = "latency-debug-window",
            .visible = this->config.debugLatencyVisible,
        });

        auto menubarConfig = std::move(this->config.menubar);
        auto onHeight = std::move(menubarConfig.onHeight);
        menubarConfig.onHeight = [this, onHeight = std::move(onHeight)](F32 height) {
            menuBarHeight = height;
            if (onHeight) {
                onHeight(height);
            }
        };
        menubar.update(std::move(menubarConfig));

        flowgraphOrder.clear();
        std::unordered_set<std::string> activeFlowgraphs;
        for (auto& flowgraphConfig : this->config.flowgraphs) {
            flowgraphOrder.push_back(flowgraphConfig.id);
            activeFlowgraphs.insert(flowgraphConfig.id);
            flowgraphConfig.dockId = static_cast<U64>(Sakura::DockspaceId());
            flowgraphWindows[flowgraphConfig.id].update(std::move(flowgraphConfig));
        }

        std::vector<std::string> staleFlowgraphs;
        for (const auto& [flowgraphId, _] : flowgraphWindows) {
            if (!activeFlowgraphs.contains(flowgraphId)) {
                staleFlowgraphs.push_back(flowgraphId);
            }
        }
        for (const auto& flowgraphId : staleFlowgraphs) {
            flowgraphWindows.erase(flowgraphId);
        }

        modal.update(std::move(this->config.modal));
    }

    void render(const Sakura::Context& ctx) {
        if (config.filePending) {
            workspaceBackground.render(ctx);
            filePendingHud.render(ctx, [this](const Sakura::Context& ctx) {
                filePendingText.render(ctx);
            });
            return;
        }

        infoHud.render(ctx);
        welcomeHud.render(ctx);
        remoteHud.render(ctx);
        notifications.render(ctx);
        debugWindow.render(ctx);

        menubar.render(ctx);
        workspaceBackground.render(ctx);
        Sakura::Dockspace({.topOffset = Sakura::Scale(ctx, menuBarHeight)});

        for (const auto& flowgraphId : flowgraphOrder) {
            if (flowgraphWindows.contains(flowgraphId)) {
                flowgraphWindows.at(flowgraphId).render(ctx);
            }
        }

        modal.render(ctx);
    }

 private:
    Config config;
    F32 menuBarHeight = 0.0f;
    Sakura::Hud filePendingHud;
    Sakura::Text filePendingText;
    Sakura::WorkspaceBackground workspaceBackground;
    InfoHudView infoHud;
    WelcomeHudView welcomeHud;
    RemoteHudView remoteHud;
    Sakura::Notifications notifications;
    Sakura::DebugWindow debugWindow;
    MenubarView menubar;
    std::vector<std::string> flowgraphOrder;
    std::unordered_map<std::string, FlowgraphWindow> flowgraphWindows;
    ModalView modal;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_WORKBENCH_HH
