#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_WINDOW_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_WINDOW_HH

#include "editor/base.hh"

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/flowgraph/window.hh"

#include "jetstream/flowgraph.hh"

#include <memory>
#include <string>

namespace Jetstream {

struct FlowgraphWindowPresenter {
    const PresenterContext& context;
    FlowgraphEditorPresenter editor;

    explicit FlowgraphWindowPresenter(const PresenterContext& context) : context(context),
                                                                         editor(context) {}

    FlowgraphWindow::Config build(const std::string& flowgraphId,
                                  const std::shared_ptr<Flowgraph>& flowgraph) const {
        const auto enqueue = context.callbacks.enqueueMail;
        const auto blocks = flowgraph->blockList();
        return FlowgraphWindow::Config{
            .id = flowgraphId,
            .title = flowgraph->title(),
            .editor = editor.build(flowgraphId, flowgraph),
            .empty = blocks.empty(),
            .onSave = [enqueue, flowgraphId]() {
                enqueue(MailSaveFlowgraph{.flowgraph = flowgraphId});
            },
            .onClose = [enqueue, flowgraphId]() {
                enqueue(MailCloseFlowgraph{flowgraphId});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_WINDOW_HH
