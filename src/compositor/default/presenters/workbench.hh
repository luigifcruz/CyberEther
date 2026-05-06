#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_WORKBENCH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_WORKBENCH_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"
#include "../themes.hh"
#include "../views/workbench.hh"

#include "jetstream/config.hh"

#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct DefaultWorkbenchPresenter {
    using ModalContent = DefaultCompositorState::ModalState::Content;
    using SettingsSection = DefaultCompositorState::SettingsState::Section;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    DefaultWorkbenchPresenter(DefaultCompositorState& state,
                              DefaultCompositorCallbacks& callbacks) :
        state(state),
        callbacks(callbacks) {}

    InfoHudView::Config buildInfoHud() const {
        return InfoHudView::Config{
            .visible = state.interface.infoPanelEnabled,
            .frameRate = Sakura::FrameRate(),
            .viewportName = state.system.viewport->name(),
            .renderInfo = state.system.render->info(),
        };
    }

    WelcomeHudView::Config buildWelcomeHud() const {
        const auto enqueue = callbacks.enqueueMail;
        return WelcomeHudView::Config{
            .visible = state.flowgraph.items.empty() && !state.modal.content.has_value(),
            .version = "v" JETSTREAM_VERSION_STR,
            .onNewFlowgraph = [enqueue]() {
                enqueue(MailNewFlowgraph{});
            },
            .onOpenFlowgraph = [enqueue]() {
                enqueue(MailOpenFlowgraph{});
            },
            .onOpenExamples = [enqueue]() {
                enqueue(MailOpenModal{.content = ModalContent::FlowgraphExamples});
            },
            .onOpenWebsite = [enqueue]() {
                enqueue(MailOpenUrl{.url = "https://cyberether.org"});
            },
            .onOpenDocs = [enqueue]() {
                enqueue(MailOpenUrl{.url = "https://cyberether.org/docs"});
            },
            .onOpenSettings = [enqueue]() {
                enqueue(MailOpenModal{.content = ModalContent::Settings, .settings = SettingsSection::General});
            },
            .onOpenAbout = [enqueue]() {
                enqueue(MailOpenModal{.content = ModalContent::About});
            },
        };
    }

    RemoteHudView::Config buildRemoteHud() const {
        const auto enqueue = callbacks.enqueueMail;
        return RemoteHudView::Config{
            .visible = state.remote.started,
            .clientCount = state.remote.started ? state.remote.clientCount : 0,
            .onOpen = [enqueue]() {
                enqueue(MailOpenModal{.content = ModalContent::RemoteStreaming});
            },
        };
    }

    MenubarView::Config buildMenubar() const {
        const auto enqueue = callbacks.enqueueMail;
        const auto focusedFlowgraph = state.interface.focusedFlowgraph;
        const bool infoPanelEnabled = state.interface.infoPanelEnabled;
        const bool backgroundParticles = state.interface.backgroundParticles;
        const bool debugLatencyEnabled = state.debug.latencyEnabled;
        const bool debugRuntimeMetricsEnabled = state.debug.runtimeMetricsEnabled;

        return MenubarView::Config{
            .id = "main-menubar",
            .hasFocusedFlowgraph = focusedFlowgraph.has_value(),
            .infoPanelEnabled = infoPanelEnabled,
            .backgroundParticles = backgroundParticles,
            .remoteSupported = state.remote.supported,
            .debugLatencyEnabled = debugLatencyEnabled,
            .debugRuntimeMetricsEnabled = debugRuntimeMetricsEnabled,
            .debugLogLevel = state.debug.logLevel,
            .themes = BuildThemeKeys(),
            .currentThemeKey = state.sakura.themeKey,
            .onAction = [enqueue,
                         focusedFlowgraph,
                         infoPanelEnabled,
                         backgroundParticles,
                         debugLatencyEnabled,
                         debugRuntimeMetricsEnabled](const MenubarView::Action action) {
                switch (action) {
                    case MenubarView::Action::About:
                        enqueue(MailOpenModal{.content = ModalContent::About});
                        break;
                    case MenubarView::Action::ViewLicense:
                    case MenubarView::Action::ViewThirdPartyOss:
                        enqueue(MailOpenModal{.content = ModalContent::Settings, .settings = SettingsSection::Legal});
                        break;
                    case MenubarView::Action::CheckForUpdates:
                        enqueue(MailCheckForUpdates{});
                        break;
                    case MenubarView::Action::Preferences:
                        enqueue(MailOpenModal{.content = ModalContent::Settings, .settings = SettingsSection::General});
                        break;
                    case MenubarView::Action::Quit:
                        enqueue(MailQuit{});
                        break;
                    case MenubarView::Action::NewFlowgraph:
                        enqueue(MailNewFlowgraph{});
                        break;
                    case MenubarView::Action::OpenFlowgraph:
                        enqueue(MailOpenFlowgraph{});
                        break;
                    case MenubarView::Action::SaveFlowgraph:
                        if (focusedFlowgraph.has_value()) {
                            enqueue(MailSaveFlowgraph{.flowgraph = focusedFlowgraph.value()});
                        } else {
                            enqueue(MailNotify{.type = Sakura::NotificationType::Error,
                                               .durationMs = 5000,
                                               .message = "No focused flowgraph to save."});
                        }
                        break;
                    case MenubarView::Action::ShowFlowgraphInfo:
                        if (focusedFlowgraph.has_value()) {
                            enqueue(MailOpenModal{
                                .content = ModalContent::FlowgraphInfo,
                                .flowgraph = focusedFlowgraph.value(),
                            });
                        } else {
                            enqueue(MailNotify{.type = Sakura::NotificationType::Error,
                                               .durationMs = 5000,
                                               .message = "No focused flowgraph to display information."});
                        }
                        break;
                    case MenubarView::Action::CloseFlowgraph:
                        if (focusedFlowgraph.has_value()) {
                            enqueue(MailCloseFlowgraph{focusedFlowgraph.value()});
                        } else {
                            enqueue(MailNotify{.type = Sakura::NotificationType::Error,
                                               .durationMs = 5000,
                                               .message = "No focused flowgraph to close."});
                        }
                        break;
                    case MenubarView::Action::RenameFlowgraph:
                        if (focusedFlowgraph.has_value()) {
                            enqueue(MailOpenModal{
                                .content = ModalContent::FlowgraphInfo,
                                .flowgraph = focusedFlowgraph.value(),
                            });
                        }
                        break;
                    case MenubarView::Action::OpenExamples:
                        enqueue(MailOpenModal{.content = ModalContent::FlowgraphExamples});
                        break;
                    case MenubarView::Action::ToggleInfoPanel:
                        enqueue(MailSetInfoPanelEnabled{.value = !infoPanelEnabled});
                        break;
                    case MenubarView::Action::ToggleBackgroundParticles:
                        enqueue(MailSetBackgroundParticles{.value = !backgroundParticles});
                        break;
                    case MenubarView::Action::RemoteStreaming:
                        enqueue(MailOpenModal{.content = ModalContent::RemoteStreaming});
                        break;
                    case MenubarView::Action::OpenSettings:
                        enqueue(MailOpenModal{.content = ModalContent::Settings, .settings = SettingsSection::General});
                        break;
                    case MenubarView::Action::ToggleDebugLatency:
                        enqueue(MailSetDebugLatencyEnabled{.value = !debugLatencyEnabled});
                        break;
                    case MenubarView::Action::ToggleRuntimeMetrics:
                        enqueue(MailSetDebugRuntimeMetricsEnabled{.value = !debugRuntimeMetricsEnabled});
                        break;
                    case MenubarView::Action::ShowBenchmarks:
                        enqueue(MailOpenModal{.content = ModalContent::Benchmark});
                        break;
                    case MenubarView::Action::OpenDeveloperSettings:
                        enqueue(MailOpenModal{.content = ModalContent::Settings, .settings = SettingsSection::Developer});
                        break;
                    case MenubarView::Action::GettingStarted:
                    case MenubarView::Action::Documentation:
                    case MenubarView::Action::OpenRepository:
                        enqueue(MailOpenUrl{.url = "https://github.com/luigifcruz/CyberEther", .notifyResult = true});
                        break;
                    case MenubarView::Action::OpenTwitter:
                        enqueue(MailOpenUrl{.url = "https://twitter.com/luigifcruz", .notifyResult = true});
                        break;
                    case MenubarView::Action::ReportIssue:
                        enqueue(MailOpenUrl{.url = "https://github.com/luigifcruz/CyberEther/issues", .notifyResult = true});
                        break;
                }
            },
            .onThemeSelect = [enqueue](const std::string& themeKey) {
                enqueue(MailApplyTheme{themeKey});
            },
            .onDebugLogLevelSelect = [enqueue](const I32 logLevel) {
                enqueue(MailSetDebugLogLevel{.value = logLevel});
            },
        };
    }

    WorkbenchView::Config build(std::vector<FlowgraphWindow::Config> flowgraphs,
                                ModalView::Config modal) const {
        WorkbenchView::Config config;
        config.filePending = state.interface.filePending;
        config.backgroundParticles = state.interface.backgroundParticles;
        config.debugLatencyVisible = state.debug.latencyEnabled;
        config.infoHud = buildInfoHud();
        config.welcomeHud = buildWelcomeHud();
        config.remoteHud = buildRemoteHud();
        config.menubar = buildMenubar();
        config.flowgraphs = std::move(flowgraphs);
        config.modal = std::move(modal);
        return config;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_WORKBENCH_HH
