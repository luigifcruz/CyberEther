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

    explicit WorkbenchPresenter(const PresenterContext& context) : context(context),
                                                                   filePicker(context),
                                                                   titlebar(context),
                                                                   welcomeHud(context),
                                                                   flowgraphPane(context),
                                                                   flowgraphMetadata(context),
                                                                   flowgraphEnvironment(context),
                                                                   modal(context) {}

    WorkbenchView::Config build() const {
        WorkbenchView::Config config;
        config.filePending = context.state.interface.filePending;
        config.titlebar = titlebar.build();
        config.welcomeHud = welcomeHud.build();
        if (auto welcome = welcomeHud.buildStartTab()) {
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
        config.backgroundParticles = context.state.interface.backgroundParticles;
        return config;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_WORKBENCH_HH
