#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_NODE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_NODE_HH

#include "picker.hh"
#include "surface.hh"
#include "../config.hh"

#include "../../context.hh"

#include "../../../model/meta.hh"
#include "../../../views/flowgraph/editor/base.hh"

#include "jetstream/block.hh"
#include "jetstream/block_interface.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/flowgraph_view.hh"
#include "jetstream/parser.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/runtime_context.hh"

#include <algorithm>
#include <any>
#include <memory>
#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphNodePresenter {
    const PresenterContext& context;
    FlowgraphCatalogPresenter catalog;
    FlowgraphSurfacePresenter surfaces;

    explicit FlowgraphNodePresenter(const PresenterContext& context);
    void buildInputs(FlowgraphNode::BlockData& block, const Flowgraph::View::BlockData& blockData) const;
    void buildOutputs(FlowgraphNode::BlockData& block, const Flowgraph::View::BlockData& blockData) const;
    void buildConfigFields(FlowgraphNode::BlockData& block,
                           const std::string& flowgraphId,
                           const std::string& blockName,
                           const std::string& nodeViewId,
                           const Flowgraph::View::BlockData& blockData) const;
    void buildTiming(FlowgraphNode::BlockData& block,
                     const Flowgraph::View::BlockData& blockData) const;
    FlowgraphNode::BlockData build(const std::string& flowgraphId,
                                   const std::shared_ptr<Flowgraph>& flowgraph,
                                   const std::string& blockName,
                                   const Flowgraph::View::BlockData& blockData) const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_NODE_HH
