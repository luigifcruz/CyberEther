#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MENUBAR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MENUBAR_HH

#include "context.hh"

#include "../model/messages.hh"
#include "../themes.hh"
#include "../views/menubar.hh"

#include <string>

namespace Jetstream {

struct MenubarPresenter {
    const PresenterContext& context;

    explicit MenubarPresenter(const PresenterContext& context) : context(context) {}

    MenubarView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        const auto focusedFlowgraph = context.state.interface.focusedFlowgraph;
        const bool infoPanelEnabled = context.state.interface.infoPanelEnabled;
        const bool backgroundParticles = context.state.interface.backgroundParticles;
        const bool flowgraphMetadataVisible = context.state.interface.flowgraphMetadataVisible;
        const bool flowgraphEnvironmentVisible = context.state.interface.flowgraphEnvironmentVisible;
        const bool debugLatencyEnabled = context.state.debug.latencyEnabled;
        const bool debugTimingEnabled = context.state.debug.timingEnabled;

        return MenubarView::Config{
            .id = "main-menubar",
            .hasFocusedFlowgraph = focusedFlowgraph.has_value(),
            .infoPanelEnabled = infoPanelEnabled,
            .backgroundParticles = backgroundParticles,
            .flowgraphMetadataVisible = flowgraphMetadataVisible,
            .flowgraphEnvironmentVisible = flowgraphEnvironmentVisible,
            .remoteSupported = context.state.remote.supported,
            .debugLatencyEnabled = debugLatencyEnabled,
            .debugTimingEnabled = debugTimingEnabled,
            .debugLogLevel = context.state.debug.logLevel,
            .themes = BuildThemeKeys(),
            .currentThemeKey = context.state.sakura.themeKey,
            .onAction = [enqueue,
                         focusedFlowgraph,
                         infoPanelEnabled,
                         backgroundParticles,
                         flowgraphMetadataVisible,
                         flowgraphEnvironmentVisible,
                         debugLatencyEnabled,
                         debugTimingEnabled](const MenubarView::Action action) {
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
                            enqueue(MailNotify{.type = Sakura::ToastType::Error,
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
                            enqueue(MailNotify{.type = Sakura::ToastType::Error,
                                               .durationMs = 5000,
                                               .message = "No focused flowgraph to display information."});
                        }
                        break;
                    case MenubarView::Action::ToggleFlowgraphMetadata:
                        if (focusedFlowgraph.has_value()) {
                            enqueue(MailSetFlowgraphMetadataVisible{.value = !flowgraphMetadataVisible});
                        } else {
                            enqueue(MailNotify{.type = Sakura::ToastType::Error,
                                               .durationMs = 5000,
                                               .message = "No focused flowgraph to display metadata."});
                        }
                        break;
                    case MenubarView::Action::ToggleFlowgraphEnvironment:
                        if (focusedFlowgraph.has_value()) {
                            enqueue(MailSetFlowgraphEnvironmentVisible{.value = !flowgraphEnvironmentVisible});
                        } else {
                            enqueue(MailNotify{.type = Sakura::ToastType::Error,
                                               .durationMs = 5000,
                                               .message = "No focused flowgraph to display environment."});
                        }
                        break;
                    case MenubarView::Action::CloseFlowgraph:
                        if (focusedFlowgraph.has_value()) {
                            enqueue(MailCloseFlowgraph{focusedFlowgraph.value()});
                        } else {
                            enqueue(MailNotify{.type = Sakura::ToastType::Error,
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
                    case MenubarView::Action::ReloadPlugins:
                        enqueue(MailReloadAllPlugins{});
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
                    case MenubarView::Action::ToggleTiming:
                        enqueue(MailSetDebugTimingEnabled{.value = !debugTimingEnabled});
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
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MENUBAR_HH
