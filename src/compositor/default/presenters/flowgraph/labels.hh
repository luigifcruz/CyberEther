#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_LABELS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_LABELS_HH

#include "jetstream/flowgraph.hh"

#include <memory>
#include <string>

namespace Jetstream {

std::string MakeFlowgraphTitle(const std::string& flowgraphId,
                               const std::shared_ptr<Flowgraph>& flowgraph);

std::string MakeSurfaceTileTitle(const std::string& blockName, const std::string& blockTitle);

std::string MakeConfigTileTitle(const std::string& blockName, const std::string& blockTitle);

std::string MakeDocumentationTileTitle(const std::string& blockName, const std::string& blockTitle);

std::string MakeEditorPaneId(const std::string& flowgraphId);

std::string MakeFlowgraphLayoutGroup(const std::string& flowgraphId);

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_LABELS_HH
