#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_SURFACE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_SURFACE_HH

#include "labels.hh"
#include "layout.hh"
#include "config.hh"
#include "surface_callbacks.hh"

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../model/meta.hh"
#include "../../views/flowgraph/surface.hh"

#include "jetstream/block.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/flowgraph_view.hh"
#include "jetstream/parser.hh"
#include "jetstream/runtime_context.hh"

#include <any>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphDetachedSurfacePresenter {
    const PresenterContext& context;

    explicit FlowgraphDetachedSurfacePresenter(const PresenterContext& context);
    std::vector<FlowgraphDetachedSurface::Config> build(const std::string& flowgraphId,
                                                        const std::shared_ptr<Flowgraph>& flowgraph) const;

};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_SURFACE_HH
