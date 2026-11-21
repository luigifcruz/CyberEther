#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_TARGET_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_TARGET_HH

#include "../../context.hh"

#include <optional>
#include <string>

namespace Jetstream {

std::optional<std::string> BuildTargetFlowgraphId(const PresenterContext& context);

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_TARGET_HH
