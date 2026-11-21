#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_CLIPBOARD_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_CLIPBOARD_HH

#include "../model/history.hh"

#include "jetstream/flowgraph.hh"

#include <optional>
#include <string>
#include <vector>

namespace Jetstream {

FlowgraphSnapshot CopyFlowgraphBlocks(const FlowgraphSnapshot& source, const std::vector<std::string>& names);

Result EncodeFlowgraphBlocks(const FlowgraphSnapshot& blocks, std::string& yaml);

std::optional<FlowgraphSnapshot> DecodeFlowgraphBlocks(const std::string& yaml);

Result PasteFlowgraphBlocks(Flowgraph& flowgraph,
                            const FlowgraphSnapshot& blocks,
                            std::optional<Extent2D<F32>> point,
                            std::vector<std::string>& created);

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_CLIPBOARD_HH
