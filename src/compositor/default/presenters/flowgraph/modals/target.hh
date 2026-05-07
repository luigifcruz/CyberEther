#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_TARGET_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_TARGET_HH

#include "../../context.hh"

#include <optional>
#include <string>

namespace Jetstream {

inline std::optional<std::string> BuildTargetFlowgraphId(const PresenterContext& context) {
    return context.state.modal.flowgraph.has_value()
        ? context.state.modal.flowgraph
        : context.state.interface.focusedFlowgraph;
}

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_TARGET_HH
