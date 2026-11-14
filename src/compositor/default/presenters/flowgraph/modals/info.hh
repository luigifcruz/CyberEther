#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_INFO_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_INFO_HH

#include "target.hh"

#include "../../../model/messages.hh"
#include "../../../views/flowgraph/modals/info.hh"

#include <optional>
#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphInfoModalPresenter {
    const PresenterContext& context;

    explicit FlowgraphInfoModalPresenter(const PresenterContext& context);
    std::optional<FlowgraphInfoView::Config> build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_INFO_HH
