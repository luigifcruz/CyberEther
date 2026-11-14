#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_HISTORY_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_HISTORY_HH

#include "../model/history.hh"

#include "jetstream/flowgraph.hh"

#include <any>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

enum class FlowgraphChange {
    None,
    Layout,
    Content,
};

struct FlowgraphRestoreReport {
    std::vector<std::pair<std::string, std::string>> renamed;
    std::vector<std::string> removed;
    FlowgraphHistory::Entry reached;
};

std::size_t FlowgraphValueFingerprint(const std::any& value);

std::vector<const FlowgraphSnapshot::Block*> FlowgraphDependencyOrder(const FlowgraphSnapshot& snapshot);

FlowgraphHistory::Entry CaptureFlowgraphSnapshot(const Flowgraph& flowgraph);

FlowgraphChange ClassifyFlowgraphChange(const FlowgraphSnapshot& previous, const FlowgraphSnapshot& next);

Result CreateFlowgraphBlock(Flowgraph& flowgraph, const FlowgraphSnapshot::Block& block);

Result ApplyFlowgraphSnapshot(Flowgraph& flowgraph,
                              const FlowgraphSnapshot& target,
                              FlowgraphRestoreReport& report);

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_HISTORY_HH
