#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_TITLEBAR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_TITLEBAR_HH

#include "context.hh"

#include "../model/messages.hh"
#include "../views/titlebar.hh"

#include "render/sakura/titlebar.hh"

#include <memory>
#include <string>

namespace Jetstream {

struct TitlebarPresenter {
    const PresenterContext& context;

    explicit TitlebarPresenter(const PresenterContext& context) : context(context) {}

    TitlebarView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        const auto focusedFlowgraph = context.state.interface.focusedFlowgraph;
        const bool startTabFocused = context.state.startTabActive();
        const bool settingsTabFocused = context.state.settingsTabActive();
        const bool docsTabFocused = context.state.docsTabActive();
        const bool flowgraphMetadataVisible = context.state.interface.flowgraphMetadataVisible;
        const bool flowgraphEnvironmentVisible = context.state.interface.flowgraphEnvironmentVisible;
        const bool canCheckForUpdates = context.state.update.supported &&
                                        !context.state.update.checking &&
                                        !context.state.update.available &&
                                        !context.state.update.downloading &&
                                        !context.state.update.ready &&
                                        !context.state.update.applying;
        const std::shared_ptr<Render::Window> render = context.state.system.render;
        const Sakura::Titlebar::Chrome chrome = Sakura::Titlebar::Resolve(render.get());

        const auto missing = [enqueue](const std::string& message) {
            enqueue(MailNotify{.type = Sakura::ToastType::Error, .durationMs = 5000, .message = message});
        };

        return TitlebarView::Config{
            .id = "main-titlebar",
            .leadingInset = chrome.leadingInset,
            .height = chrome.height,
            .remoteSupported = context.state.remote.supported,
            .remoteStarted = context.state.remote.started,
            .updateAvailable = context.state.update.available,
            .updateReady = context.state.update.ready,
            .intelligenceEnabled = context.state.agent.settings.enabled,
            .agentVisible = context.state.agent.ui.visible,
            .voiceVisible = context.state.agent.ui.voiceVisible,
            .focusedFlowgraph = focusedFlowgraph,
            .flowgraphMetadataVisible = flowgraphMetadataVisible,
            .flowgraphEnvironmentVisible = flowgraphEnvironmentVisible,
            .debugTimingEnabled = context.state.debug.timingEnabled,
            .debugLogLevel = context.state.debug.logLevel,
            .onAction = [enqueue, focusedFlowgraph, startTabFocused, settingsTabFocused, docsTabFocused,
                         canCheckForUpdates, missing,
                         timing = context.state.debug.timingEnabled,
                         agentVisible = context.state.agent.ui.visible,
                         voiceVisible = context.state.agent.ui.voiceVisible](TitlebarView::Action action) {
                using Action = TitlebarView::Action;
                switch (action) {
                    case Action::About:
                        enqueue(MailOpenModal{.content = ModalContent::About});
                        break;
                    case Action::Settings:
                        enqueue(MailOpenSettings{.section = SettingsSection::General});
                        break;
                    case Action::CheckForUpdates:
                        enqueue(MailOpenSettings{.section = SettingsSection::About});
                        if (canCheckForUpdates) {
                            enqueue(MailCheckForUpdates{});
                        }
                        break;
                    case Action::OpenUpdateModal:
                        enqueue(MailOpenSettings{.section = SettingsSection::About});
                        break;
                    case Action::RemoteStreaming:
                        enqueue(MailOpenStartTab{});
                        break;
                    case Action::ToggleAgent:
                        enqueue(MailSetAgentChatVisible{.value = !agentVisible});
                        break;
                    case Action::ToggleVoice:
                        enqueue(MailSetAgentVoiceVisible{.value = !voiceVisible});
                        break;
                    case Action::GettingStarted:
                        enqueue(MailOpenDocs{.page = "quick-start"});
                        break;
                    case Action::Documentation:
                        enqueue(MailOpenDocs{});
                        break;
                    case Action::OpenRepository:
                        enqueue(MailOpenUrl{.url = "https://github.com/luigifcruz/CyberEther", .notifyResult = true});
                        break;
                    case Action::ReportIssue:
                        enqueue(MailOpenUrl{.url = "https://github.com/luigifcruz/CyberEther/issues", .notifyResult = true});
                        break;
                    case Action::OpenTwitter:
                        enqueue(MailOpenUrl{.url = "https://twitter.com/luigifcruz", .notifyResult = true});
                        break;
                    case Action::SendFeedback:
                        enqueue(MailOpenModal{.content = ModalContent::Feedback});
                        break;
                    case Action::ToggleTiming:
                        enqueue(MailSetDebugTimingEnabled{.value = !timing});
                        break;
                    case Action::ShowBenchmarks:
                        enqueue(MailOpenModal{.content = ModalContent::Benchmark});
                        break;
                    case Action::ReloadPlugins:
                        enqueue(MailReloadAllPlugins{});
                        break;
                    case Action::OpenDeveloperSettings:
                        enqueue(MailOpenSettings{.section = SettingsSection::Developer});
                        break;
                    case Action::NewTab:
                        enqueue(MailOpenStartTab{});
                        break;
                    case Action::Quit:
                        enqueue(MailQuit{});
                        break;
                    case Action::NewFlowgraph:
                        enqueue(MailNewFlowgraph{});
                        break;
                    case Action::OpenFlowgraph:
                        enqueue(MailOpenFlowgraph{});
                        break;
                    case Action::SaveFlowgraph:
                        if (focusedFlowgraph.has_value()) {
                            enqueue(MailSaveFlowgraph{.flowgraph = focusedFlowgraph.value()});
                        } else {
                            missing("No focused flowgraph to save.");
                        }
                        break;
                    case Action::ShowFlowgraphInfo:
                        if (focusedFlowgraph.has_value()) {
                            enqueue(MailOpenModal{
                                .content = ModalContent::FlowgraphInfo,
                                .flowgraph = focusedFlowgraph.value(),
                            });
                        } else {
                            missing("No focused flowgraph to display information.");
                        }
                        break;
                    case Action::CloseFlowgraph:
                        if (startTabFocused) {
                            enqueue(MailCloseStartTab{});
                        } else if (settingsTabFocused) {
                            enqueue(MailCloseSettings{});
                        } else if (docsTabFocused) {
                            enqueue(MailCloseDocs{});
                        } else if (focusedFlowgraph.has_value()) {
                            enqueue(MailCloseFlowgraph{focusedFlowgraph.value()});
                        } else {
                            missing("No focused flowgraph to close.");
                        }
                        break;
                }
            },
            .onLogLevel = [enqueue](I32 level) {
                enqueue(MailSetDebugLogLevel{.value = level});
            },
            .onTabAction = [enqueue,
                            focusedFlowgraph,
                            flowgraphMetadataVisible,
                            flowgraphEnvironmentVisible](TitlebarView::TabAction action,
                                                         const std::string& flowgraph) {
                using TabAction = TitlebarView::TabAction;
                const bool focused = focusedFlowgraph == flowgraph;
                switch (action) {
                    case TabAction::Save:
                        enqueue(MailSaveFlowgraph{.flowgraph = flowgraph});
                        break;
                    case TabAction::Rename:
                    case TabAction::Info:
                        enqueue(MailOpenModal{.content = ModalContent::FlowgraphInfo, .flowgraph = flowgraph});
                        break;
                    case TabAction::ToggleMetadata:
                        if (!focused) {
                            enqueue(MailFocusFlowgraph{.flowgraph = flowgraph});
                        }
                        enqueue(MailSetFlowgraphMetadataVisible{.value = !focused || !flowgraphMetadataVisible});
                        break;
                    case TabAction::ToggleEnvironment:
                        if (!focused) {
                            enqueue(MailFocusFlowgraph{.flowgraph = flowgraph});
                        }
                        enqueue(MailSetFlowgraphEnvironmentVisible{.value = !focused || !flowgraphEnvironmentVisible});
                        break;
                    case TabAction::Close:
                        enqueue(MailCloseFlowgraph{flowgraph});
                        break;
                }
            },
            .onBarPress = [render](U8 clicks) {
                Sakura::Titlebar::Press(render.get(), clicks);
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_TITLEBAR_HH
