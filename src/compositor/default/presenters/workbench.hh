#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_WORKBENCH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_WORKBENCH_HH

#include "context.hh"
#include "docs.hh"
#include "file_picker.hh"
#include "flowgraph/close_prompt.hh"
#include "flowgraph/pane.hh"
#include "modal/container.hh"
#include "settings/base.hh"
#include "start_tab/base.hh"
#include "terminal.hh"
#include "titlebar.hh"

#include "agent/chat.hh"
#include "../views/workbench.hh"

namespace Jetstream {

struct WorkbenchPresenter {
    const PresenterContext& context;
    FilePickerPresenter filePicker;
    TitlebarPresenter titlebar;
    StartTabPresenter startTab;
    FlowgraphPanePresenter flowgraphPane;
    FlowgraphClosePromptPresenter closePrompt;
    ModalPresenter modal;
    SettingsPresenter settings;
    DocsPresenter docs;
    AgentChatPresenter agentChat;
    TerminalPresenter terminals;

    explicit WorkbenchPresenter(const PresenterContext& context);
    WorkbenchView::Config build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_WORKBENCH_HH
