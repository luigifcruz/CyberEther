#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_AREA_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_AREA_HH

#include "window.hh"

#include "../context.hh"

#include "../../views/flowgraph/window.hh"

#include <vector>

namespace Jetstream {

struct FlowgraphAreaPresenter {
    const PresenterContext& context;
    FlowgraphWindowPresenter window;

    explicit FlowgraphAreaPresenter(const PresenterContext& context) : context(context),
                                                                       window(context) {}

    std::vector<FlowgraphWindow::Config> build() const {
        std::vector<FlowgraphWindow::Config> configs;
        configs.reserve(context.state.flowgraph.items.size());

        for (const auto& [flowgraphId, flowgraph] : context.state.flowgraph.items) {
            configs.push_back(window.build(flowgraphId, flowgraph));
        }

        return configs;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_AREA_HH
