#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_EXAMPLES_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_EXAMPLES_HH

#include "../../context.hh"

#include "../../../model/messages.hh"
#include "../../../views/flowgraph/modals/examples.hh"

#include "jetstream/registry.hh"

#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphExamplesModalPresenter {
    const PresenterContext& context;

    explicit FlowgraphExamplesModalPresenter(const PresenterContext& context) : context(context) {}

    FlowgraphExamplesView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        std::vector<FlowgraphExamplesView::Example> examples;
        for (const auto& entry : Registry::ListAvailableFlowgraphs()) {
            examples.push_back({
                .key = entry.key,
                .title = entry.title,
                .summary = entry.summary,
                .content = entry.content,
            });
        }

        return FlowgraphExamplesView::Config{
            .examples = std::move(examples),
            .onOpen = [enqueue](const FlowgraphExamplesView::Example& flowgraph) {
                std::vector<char> blob(flowgraph.content.begin(), flowgraph.content.end());
                enqueue(MailOpenFlowgraphBlob{std::move(blob)});
                enqueue(MailCloseModal{});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_EXAMPLES_HH
