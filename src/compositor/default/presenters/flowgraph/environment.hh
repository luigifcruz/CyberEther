#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_ENVIRONMENT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_ENVIRONMENT_HH

#include "../context.hh"
#include "key_value.hh"

#include "../../model/messages.hh"
#include "../../model/tiles.hh"
#include "../../views/flowgraph/environment.hh"

#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_environment.hh"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream {

struct FlowgraphEnvironmentPresenter {
    const PresenterContext& context;

    explicit FlowgraphEnvironmentPresenter(const PresenterContext& context);
    std::optional<FlowgraphEnvironment::Config> build(const std::string& flowgraphId,
                                                      const std::shared_ptr<Flowgraph>& flowgraph) const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_ENVIRONMENT_HH
