#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_WORKBENCH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_WORKBENCH_HH

#include "context.hh"
#include "file_picker.hh"
#include "flowgraph/environment.hh"
#include "flowgraph/metadata.hh"
#include "flowgraph/pane.hh"
#include "hud/welcome.hh"
#include "modal/container.hh"
#include "titlebar.hh"

#include "agent/chat.hh"
#include "agent/voice.hh"
#include "../views/workbench.hh"

namespace Jetstream {

struct WorkbenchPresenter {
    const PresenterContext& context;
    FilePickerPresenter filePicker;
    TitlebarPresenter titlebar;
    WelcomeHudPresenter welcomeHud;
    FlowgraphPanePresenter flowgraphPane;
    FlowgraphMetadataWindowPresenter flowgraphMetadata;
    FlowgraphEnvironmentWindowPresenter flowgraphEnvironment;
    ModalPresenter modal;
    AgentChatPresenter agentChat;
    AgentVoicePresenter agentVoice;

    explicit WorkbenchPresenter(const PresenterContext& context) : context(context),
                                                                   filePicker(context),
                                                                   titlebar(context),
                                                                   welcomeHud(context),
                                                                   flowgraphPane(context),
                                                                   flowgraphMetadata(context),
                                                                   flowgraphEnvironment(context),
                                                                   modal(context),
                                                                   agentChat{context},
                                                                   agentVoice{context} {}

    WorkbenchView::Config build() const {
        WorkbenchView::Config config;
        config.filePending = context.state.interface.filePending;
        config.titlebar = titlebar.build();
        config.agentChat = agentChat.build();
        config.agentVoice = agentVoice.build();
        config.welcomeHud = welcomeHud.build();
        if (config.welcomeHud.has_value()) {
            config.welcomeHud->chatOpen = !config.agentChat.hidden;
            config.welcomeHud->assistantEnabled = context.state.agent.settings.enabled;
        }
        if (auto welcome = welcomeHud.buildStartTab()) {
            welcome->chatOpen = !config.agentChat.hidden;
            welcome->assistantEnabled = context.state.agent.settings.enabled;
            const auto enqueue = context.callbacks.enqueueMail;
            config.startTab = StartTabView::Config{
                .welcome = std::move(*welcome),
                .onClose = [enqueue]() {
                    enqueue(MailCloseStartTab{});
                },
                .onFocus = [enqueue]() {
                    enqueue(MailFocusStartTab{});
                },
            };
        }
        config.startTabRequests = context.state.interface.startTabRequests;
        config.flowgraphs.reserve(context.state.flowgraph.items.size());
        for (const auto& [flowgraphId, flowgraph] : context.state.flowgraph.items) {
            config.flowgraphs.push_back(flowgraphPane.build(flowgraphId, flowgraph));
        }
        config.flowgraphMetadata = flowgraphMetadata.build();
        config.flowgraphEnvironment = flowgraphEnvironment.build();
        config.modal = modal.build();
        config.filePicker = filePicker.build();
        config.backgroundParticles = context.state.interface.backgroundParticles &&
                                     (config.filePending || (config.flowgraphs.empty() && !config.startTab));
        return config;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_WORKBENCH_HH
