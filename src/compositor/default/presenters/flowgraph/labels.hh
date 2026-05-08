#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_LABELS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_LABELS_HH

#include "../../model/meta.hh"

#include "jetstream/flowgraph.hh"

#include <memory>
#include <string>

namespace Jetstream {

inline std::string MakeFlowgraphWindowId(const std::string& flowgraphId) {
    return flowgraphId;
}

inline std::string MakeFlowgraphWindowTitle(const std::string& flowgraphId,
                                            const std::shared_ptr<Flowgraph>& flowgraph) {
    if (flowgraph && !flowgraph->title().empty()) {
        return flowgraph->title();
    }
    return flowgraphId;
}

inline std::string MakeFlowgraphWindowLabel(const std::string& flowgraphId,
                                            const std::shared_ptr<Flowgraph>& flowgraph) {
    return MakeFlowgraphWindowTitle(flowgraphId, flowgraph) + "###" + MakeFlowgraphWindowId(flowgraphId);
}

inline std::string MakeDetachedSurfaceWindowId(const std::string& flowgraphId,
                                               const std::string& blockName,
                                               const std::string& surfaceId) {
    return flowgraphId + ":" + blockName + ":" + surfaceId;
}

inline std::string MakeDetachedSurfaceWindowTitle(const std::string& blockName,
                                                  const std::string& blockTitle) {
    const std::string title = blockTitle.empty() ? blockName : blockTitle;
    return title + " (" + blockName + ")";
}

inline std::string MakeDetachedSurfaceWindowLabel(const std::string& flowgraphId,
                                                  const std::string& blockName,
                                                  const std::string& surfaceId,
                                                  const std::string& blockTitle) {
    return MakeDetachedSurfaceWindowTitle(blockName, blockTitle) + "###" +
           MakeDetachedSurfaceWindowId(flowgraphId, blockName, surfaceId);
}

inline std::string MakeStackWindowId(const std::string& flowgraphId, const std::string& stackId) {
    return "stack:" + flowgraphId + ":" + stackId;
}

inline std::string MakeStackWindowTitle(const std::string& stackId, const StackMeta& stackMeta) {
    return stackMeta.title.empty() ? stackId : stackMeta.title;
}

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_LABELS_HH
