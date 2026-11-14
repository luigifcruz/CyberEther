#include "close_prompt.hh"

namespace Jetstream {

FlowgraphClosePromptPresenter::FlowgraphClosePromptPresenter(const PresenterContext& context) : context(context) {}

std::optional<FlowgraphClosePromptView::Config> FlowgraphClosePromptPresenter::build() const {
    const auto& closing = context.state.interface.closingFlowgraph;
    if (!closing.has_value() || !context.state.flowgraph.items.contains(*closing)) {
        return std::nullopt;
    }
    const auto enqueue = context.callbacks.enqueueMail;
    const std::string flowgraphId = *closing;
    return FlowgraphClosePromptView::Config{
        .flowgraph = flowgraphId,
        .onSave = [enqueue, flowgraphId]() {
            enqueue(MailCancelCloseFlowgraph{});
            enqueue(MailOpenModal{
                .content = ModalContent::FlowgraphInfo,
                .flowgraph = flowgraphId,
            });
        },
        .onDontSave = [enqueue, flowgraphId]() {
            enqueue(MailCloseFlowgraph{.flowgraph = flowgraphId, .force = true});
        },
        .onCancel = [enqueue]() {
            enqueue(MailCancelCloseFlowgraph{});
        },
    };
}

}  // namespace Jetstream
