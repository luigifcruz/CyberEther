#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_LABELS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_LABELS_HH

#include "jetstream/flowgraph.hh"

#include <memory>
#include <string>

namespace Jetstream {

inline std::string MakeFlowgraphTitle(const std::string& flowgraphId,
                                      const std::shared_ptr<Flowgraph>& flowgraph) {
    if (flowgraph && !flowgraph->title().empty()) {
        return flowgraph->title();
    }
    return flowgraphId;
}

inline std::string MakeSurfaceTileTitle(const std::string& blockName, const std::string& blockTitle) {
    const std::string title = blockTitle.empty() ? blockName : blockTitle;
    return jst::fmt::format("{} ({})", title, blockName);
}

inline std::string MakeConfigTileTitle(const std::string& blockName, const std::string& blockTitle) {
    const std::string title = blockTitle.empty() ? blockName : blockTitle;
    return jst::fmt::format("{} Controls ({})", title, blockName);
}

inline std::string MakeDocumentationTileTitle(const std::string& blockName, const std::string& blockTitle) {
    const std::string title = blockTitle.empty() ? blockName : blockTitle;
    return jst::fmt::format("{} Documentation ({})", title, blockName);
}

inline std::string MakeEditorPaneId(const std::string& flowgraphId) {
    return "editor:" + flowgraphId;
}

inline std::string MakeFlowgraphLayoutGroup(const std::string& flowgraphId) {
    return "layout:" + flowgraphId;
}

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_LABELS_HH
