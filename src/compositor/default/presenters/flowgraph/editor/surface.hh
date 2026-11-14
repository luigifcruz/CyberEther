#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_SURFACE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_SURFACE_HH

#include "../../context.hh"
#include "../surface_callbacks.hh"

#include "../../../model/messages.hh"
#include "../../../model/meta.hh"
#include "../../../views/flowgraph/editor/node.hh"

#include "jetstream/block.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/flowgraph_view.hh"

#include <memory>
#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphSurfacePresenter {
    const PresenterContext& context;

    explicit FlowgraphSurfacePresenter(const PresenterContext& context);
    void buildSurfaces(FlowgraphNode::BlockData& block,
                       const std::shared_ptr<Flowgraph>& flowgraph,
                       const std::string& flowgraphId,
                       const std::string& blockName,
                       const std::string& nodeViewId,
                       const Flowgraph::View::BlockData& blockData) const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_SURFACE_HH
