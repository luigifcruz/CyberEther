#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_DETACHED_CONFIG_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_DETACHED_CONFIG_HH

#include "config.hh"
#include "labels.hh"
#include "layout.hh"

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../model/meta.hh"
#include "../../views/flowgraph/detached_config.hh"

#include "jetstream/block.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/flowgraph_view.hh"

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace Jetstream {

struct FlowgraphDetachedConfigPresenter {
    const PresenterContext& context;

    explicit FlowgraphDetachedConfigPresenter(const PresenterContext& context);
    std::vector<FlowgraphDetachedConfig::Config> build(const std::string& flowgraphId,
                                                       const std::shared_ptr<Flowgraph>& flowgraph) const;

};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_DETACHED_CONFIG_HH
