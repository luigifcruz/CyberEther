#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_WINDOW_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_WINDOW_HH

#include "editor/base.hh"
#include "labels.hh"
#include "stack.hh"
#include "surface.hh"

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/flowgraph/window.hh"

#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_view.hh"

#include <memory>
#include <string>

namespace Jetstream {

struct FlowgraphWindowPresenter {
    const PresenterContext& context;
    FlowgraphEditorPresenter editor;
    StackPresenter stacks;
    FlowgraphDetachedSurfacePresenter surfaces;

    explicit FlowgraphWindowPresenter(const PresenterContext& context) : context(context),
                                                                          editor(context),
                                                                          stacks(context),
                                                                          surfaces(context) {}

    FlowgraphWindow::Config build(const std::string& flowgraphId,
                                  const std::shared_ptr<Flowgraph>& flowgraph) const {
        const auto enqueue = context.callbacks.enqueueMail;
        return FlowgraphWindow::Config{
            .id = MakeFlowgraphWindowId(flowgraphId),
            .title = MakeFlowgraphWindowTitle(flowgraphId, flowgraph),
            .editor = editor.build(flowgraphId, flowgraph),
            .stacks = stacks.build(flowgraphId, flowgraph),
            .detachedSurfaces = surfaces.build(flowgraphId, flowgraph),
            .empty = flowgraph->view().empty(),
            .onSave = [enqueue, flowgraphId]() {
                enqueue(MailSaveFlowgraph{.flowgraph = flowgraphId});
            },
            .onClose = [enqueue, flowgraphId]() {
                enqueue(MailCloseFlowgraph{flowgraphId});
            },
            .onCreateStack = [enqueue, flowgraphId]() {
                enqueue(MailCreateStack{.flowgraph = flowgraphId});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_WINDOW_HH
