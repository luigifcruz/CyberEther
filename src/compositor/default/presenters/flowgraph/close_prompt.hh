#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_CLOSE_PROMPT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_CLOSE_PROMPT_HH

#include "../context.hh"
#include "../../model/messages.hh"
#include "../../views/flowgraph/close_prompt.hh"

#include <optional>
#include <string>

namespace Jetstream {

struct FlowgraphClosePromptPresenter {
    const PresenterContext& context;

    explicit FlowgraphClosePromptPresenter(const PresenterContext& context) : context(context) {}

    std::optional<FlowgraphClosePromptView::Config> build() const {
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
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_CLOSE_PROMPT_HH
