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

    explicit FlowgraphClosePromptPresenter(const PresenterContext& context);
    std::optional<FlowgraphClosePromptView::Config> build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_CLOSE_PROMPT_HH
