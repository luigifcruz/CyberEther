#include "labels.hh"

namespace Jetstream {

std::string MakeFlowgraphTitle(const std::string& flowgraphId,
                               const std::shared_ptr<Flowgraph>& flowgraph) {
    if (flowgraph && !flowgraph->title().empty()) {
        return flowgraph->title();
    }
    return flowgraphId;
}

std::string MakeSurfaceTileTitle(const std::string& blockName, const std::string& blockTitle) {
    const std::string title = blockTitle.empty() ? blockName : blockTitle;
    return jst::fmt::format("{} ({})", title, blockName);
}

std::string MakeConfigTileTitle(const std::string& blockName, const std::string& blockTitle) {
    const std::string title = blockTitle.empty() ? blockName : blockTitle;
    return jst::fmt::format("{} Controls ({})", title, blockName);
}

std::string MakeDocumentationTileTitle(const std::string& blockName, const std::string& blockTitle) {
    const std::string title = blockTitle.empty() ? blockName : blockTitle;
    return jst::fmt::format("{} Documentation ({})", title, blockName);
}

std::string MakeEditorPaneId(const std::string& flowgraphId) {
    return "editor:" + flowgraphId;
}

std::string MakeFlowgraphLayoutGroup(const std::string& flowgraphId) {
    return "layout:" + flowgraphId;
}

}  // namespace Jetstream
