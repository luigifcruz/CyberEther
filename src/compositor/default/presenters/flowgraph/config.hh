#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_CONFIG_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_CONFIG_HH

#include "../../model/messages.hh"
#include "../../model/state.hh"
#include "../../views/flowgraph/editor/config/types.hh"
#include "../../views/flowgraph/editor/metrics/types.hh"

#include <jetstream/flowgraph_view.hh>
#include <jetstream/runtime_context.hh>

#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

std::vector<FlowgraphMetricConfig> BuildFlowgraphMetrics(
    const std::string& viewId, const Flowgraph::View::BlockData& block);

std::vector<FlowgraphConfigFieldConfig> BuildFlowgraphConfigFields(
    const std::string& viewId, const Flowgraph::View::BlockData& block);

void ApplyLiveMarkdown(std::vector<FlowgraphConfigFieldConfig>& fields,
                       const DefaultCompositorState::FlowgraphState& flowgraphs,
                       const std::string& flowgraphId,
                       const std::string& blockName);

void ApplyConfigDrafts(std::vector<FlowgraphConfigFieldConfig>& fields,
                       const Sakura::Callback<void(Mail&&)>& enqueue,
                       const DefaultCompositorState::FlowgraphState& flowgraphs,
                       const std::string& flowgraphId,
                       const std::string& blockName);

std::vector<FlowgraphConfigFieldConfig> BuildFlowgraphDetachedConfigFields(
    const Sakura::Callback<void(Mail&&)>& enqueue,
    const DefaultCompositorState::FlowgraphState& flowgraphs,
    const std::string& flowgraphId,
    const std::string& blockName,
    const std::string& viewId,
    const Flowgraph::View::BlockData& block);

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_CONFIG_HH
