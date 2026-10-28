#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_WORKBENCH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_WORKBENCH_HH

#include "context.hh"
#include "file_picker.hh"
#include "flowgraph/environment.hh"
#include "flowgraph/metadata.hh"
#include "flowgraph/pane.hh"
#include "modal/container.hh"
#include "settings/base.hh"
#include "start_tab/base.hh"
#include "titlebar.hh"

#include "agent/chat.hh"
#include "agent/voice.hh"
#include "../views/workbench.hh"

namespace Jetstream {

struct WorkbenchPresenter {
    const PresenterContext& context;
    FilePickerPresenter filePicker;
    TitlebarPresenter titlebar;
    StartTabPresenter startTab;
    FlowgraphPanePresenter flowgraphPane;
    FlowgraphMetadataWindowPresenter flowgraphMetadata;
    FlowgraphEnvironmentWindowPresenter flowgraphEnvironment;
    ModalPresenter modal;
    SettingsPresenter settings;
    AgentChatPresenter agentChat;
    AgentVoicePresenter agentVoice;

    explicit WorkbenchPresenter(const PresenterContext& context) : context(context),
                                                                   filePicker(context),
                                                                   titlebar(context),
                                                                   startTab(context),
                                                                   flowgraphPane(context),
                                                                   flowgraphMetadata(context),
                                                                   flowgraphEnvironment(context),
                                                                   modal(context),
                                                                   settings(context),
                                                                   agentChat{context},
                                                                   agentVoice{context} {}

    WorkbenchView::Config build() const {
        WorkbenchView::Config config;
        config.filePending = context.state.interface.filePending;
        config.titlebar = titlebar.build();
        config.agentChat = agentChat.build();
        config.agentVoice = agentVoice.build();
        config.hideTabs = context.state.tabsHidden();
        config.startTab = startTab.build();
        config.startTabRequests = context.state.interface.startTabRequests;
        config.startTabFocused = context.state.startTabActive();
        if (context.state.tabShown(WorkspaceTab::Settings)) {
            config.settings = settings.build();
        }
        config.settingsTabRequests = context.state.interface.settingsTabRequests;
        config.settingsTabFocused = context.state.settingsTabActive();
        config.flowgraphs.reserve(context.state.flowgraph.items.size());
        for (const auto& [flowgraphId, flowgraph] : context.state.flowgraph.items) {
            config.flowgraphs.push_back(flowgraphPane.build(flowgraphId, flowgraph));
        }
        config.flowgraphMetadata = flowgraphMetadata.build();
        config.flowgraphEnvironment = flowgraphEnvironment.build();
        config.modal = modal.build();
        config.filePicker = filePicker.build();
        config.backgroundParticles = context.state.interface.backgroundParticles &&
                                     (config.filePending || context.state.tabsHidden());
        return config;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_WORKBENCH_HH
