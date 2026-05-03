#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_BASE_HH

#include "flowgraph.hh"
#include "modal.hh"
#include "workbench.hh"

#include <memory>
#include <utility>

namespace Jetstream {

struct DefaultPresenterRegistry {
 public:
    DefaultPresenterRegistry(DefaultCompositorState& state,
                             DefaultCompositorCallbacks& callbacks) : state(state),
                                                                      callbacks(callbacks) {
        flowgraph = std::make_shared<DefaultFlowgraphPresenter>(this->state, this->callbacks);
        modal = std::make_shared<DefaultModalPresenter>(this->state, this->callbacks);
        workbench = std::make_shared<DefaultWorkbenchPresenter>(this->state, this->callbacks);
    }

    WorkbenchView::Config build() const {
        auto flowgraphs = flowgraph->build();
        auto flowgraphModals = flowgraph->buildModalConfigs();
        return workbench->build(std::move(flowgraphs), modal->build(std::move(flowgraphModals)));
    }

 private:
    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    std::shared_ptr<DefaultFlowgraphPresenter> flowgraph;
    std::shared_ptr<DefaultModalPresenter> modal;
    std::shared_ptr<DefaultWorkbenchPresenter> workbench;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_BASE_HH
