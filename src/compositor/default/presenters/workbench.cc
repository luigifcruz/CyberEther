#include "workbench.hh"

namespace Jetstream {

WorkbenchPresenter::WorkbenchPresenter(const PresenterContext& context) : context(context),
                                                                          filePicker(context),
                                                                          titlebar(context),
                                                                          startTab(context),
                                                                          flowgraphPane(context),
                                                                          closePrompt(context),
                                                                          modal(context),
                                                                          settings(context),
                                                                          docs(context),
                                                                          agentChat{context},
                                                                          terminals(context) {}

WorkbenchView::Config WorkbenchPresenter::build() const {
    WorkbenchView::Config config;
    config.filePending = context.state.interface.filePending;
    config.titlebar = titlebar.build();
    config.closePrompt = closePrompt.build();
    config.agentChat = agentChat.build();
    config.hideTabs = context.state.tabsHidden();
    config.startTab = startTab.build();
    config.startTabRequests = context.state.interface.startTabRequests;
    config.startTabFocused = context.state.startTabActive();
    if (context.state.interface.settingsTab) {
        config.settings = settings.build();
    }
    config.settingsTabRequests = context.state.interface.settingsTabRequests;
    config.settingsTabFocused = context.state.settingsTabActive();
    if (context.state.interface.docsTab) {
        config.docs = docs.build();
    }
    config.docsTabRequests = context.state.interface.docsTabRequests;
    config.docsTabFocused = context.state.docsTabActive();
    config.flowgraphs.reserve(context.state.flowgraph.items.size());
    for (const auto& [flowgraphId, flowgraph] : context.state.flowgraph.items) {
        config.flowgraphs.push_back(flowgraphPane.build(flowgraphId, flowgraph));
    }
    config.terminals = terminals.build();
    config.terminalTabs = terminals.tabs();
    if (context.state.terminal.requested.has_value()) {
        config.terminalRequested = MakeTerminalTabKey(*context.state.terminal.requested);
    }
    config.terminalRequests = context.state.terminal.requests;
    config.terminalFocused = context.state.terminal.requested.has_value() &&
                             context.state.terminal.focused == context.state.terminal.requested;
    config.modal = modal.build();
    config.filePicker = filePicker.build();
    config.backgroundParticles = context.state.interface.backgroundParticles &&
                                 (config.filePending || config.hideTabs);
    return config;
}

}  // namespace Jetstream
