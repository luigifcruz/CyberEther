#include "target.hh"

namespace Jetstream {

std::optional<std::string> BuildTargetFlowgraphId(const PresenterContext& context) {
    return context.state.modal.flowgraph.has_value()
        ? context.state.modal.flowgraph
        : context.state.interface.focusedFlowgraph;
}

}  // namespace Jetstream
