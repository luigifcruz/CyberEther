#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_CLOSE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_CLOSE_HH

#include "target.hh"

#include "../../../model/messages.hh"
#include "../../../views/flowgraph/modals/close.hh"

#include <optional>
#include <string>

namespace Jetstream {

struct FlowgraphCloseModalPresenter {
    const PresenterContext& context;

    explicit FlowgraphCloseModalPresenter(const PresenterContext& context) : context(context) {}

    std::optional<FlowgraphCloseView::Config> build() const {
        const auto targetFlowgraph = BuildTargetFlowgraphId(context);
        if (!targetFlowgraph.has_value()) {
            return std::nullopt;
        }

        const auto enqueue = context.callbacks.enqueueMail;
        const std::string flowgraphId = targetFlowgraph.value();
        return FlowgraphCloseView::Config{
            .onSave = [enqueue, flowgraphId]() {
                enqueue(MailOpenModal{
                    .content = ModalContent::FlowgraphInfo,
                    .flowgraph = flowgraphId,
                });
            },
            .onDontSave = [enqueue, flowgraphId]() {
                enqueue(MailCloseFlowgraph{.flowgraph = flowgraphId, .force = true});
                enqueue(MailCloseModal{});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_CLOSE_HH
