#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_WORKBENCH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_WORKBENCH_HH

#include "context.hh"
#include "file_picker.hh"
#include "flowgraph/environment.hh"
#include "flowgraph/metadata.hh"
#include "flowgraph/window.hh"
#include "hud/info.hh"
#include "hud/remote.hh"
#include "hud/welcome.hh"
#include "menubar.hh"
#include "modal/container.hh"

#include "../views/workbench.hh"

namespace Jetstream {

struct WorkbenchPresenter {
    const PresenterContext& context;
    FilePickerPresenter filePicker;
    MenubarPresenter menubar;
    WelcomeHudPresenter welcomeHud;
    InfoHudPresenter infoHud;
    RemoteHudPresenter remoteHud;
    FlowgraphWindowPresenter flowgraphWindow;
    FlowgraphMetadataWindowPresenter flowgraphMetadata;
    FlowgraphEnvironmentWindowPresenter flowgraphEnvironment;
    ModalPresenter modal;

    explicit WorkbenchPresenter(const PresenterContext& context) : context(context),
                                                                   filePicker(context),
                                                                   menubar(context),
                                                                   welcomeHud(context),
                                                                   infoHud(context),
                                                                   remoteHud(context),
                                                                   flowgraphWindow(context),
                                                                   flowgraphMetadata(context),
                                                                   flowgraphEnvironment(context),
                                                                   modal(context) {}

    WorkbenchView::Config build() const {
        WorkbenchView::Config config;
        config.filePending = context.state.interface.filePending;
        config.backgroundParticles = context.state.interface.backgroundParticles;
        if (context.state.debug.latencyEnabled) {
            config.debugWindow = Sakura::DebugWindow::Config{
                .id = "latency-debug-window",
            };
        }
        config.menubar = menubar.build();
        config.welcomeHud = welcomeHud.build();
        config.infoHud = infoHud.build();
        config.remoteHud = remoteHud.build();
        config.flowgraphs.reserve(context.state.flowgraph.items.size());
        for (const auto& [flowgraphId, flowgraph] : context.state.flowgraph.items) {
            config.flowgraphs.push_back(flowgraphWindow.build(flowgraphId, flowgraph));
        }
        config.flowgraphMetadata = flowgraphMetadata.build();
        config.flowgraphEnvironment = flowgraphEnvironment.build();
        config.modal = modal.build();
        config.filePicker = filePicker.build();
        return config;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_WORKBENCH_HH
