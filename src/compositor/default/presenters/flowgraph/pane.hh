#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_PANE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_PANE_HH

#include "detached_config.hh"
#include "editor/base.hh"
#include "environment.hh"
#include "labels.hh"
#include "layout.hh"
#include "surface.hh"

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/components/callout.hh"
#include "../../views/flowgraph/pane.hh"

#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_view.hh"
#include "tools/text.hh"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphPanePresenter {
    const PresenterContext& context;
    FlowgraphEditorPresenter editor;
    LayoutPresenter layout;
    FlowgraphDetachedSurfacePresenter surfaces;
    FlowgraphDetachedConfigPresenter configs;
    FlowgraphEnvironmentPresenter environment;

    explicit FlowgraphPanePresenter(const PresenterContext& context);
    FlowgraphPane::Config build(const std::string& flowgraphId,
                                const std::shared_ptr<Flowgraph>& flowgraph) const;

 private:
    std::optional<FlowgraphPane::Description> description(const std::string& flowgraphId,
                                                          const std::shared_ptr<Flowgraph>& flowgraph,
                                                          const Sakura::Callback<void(Mail&&)>& enqueue) const;
    std::optional<FlowgraphDocumentation::Config> documentation(const std::string& flowgraphId) const;
    Callout::Tone dependencyReviewTone() const;
    std::string dependencyReviewMessage() const;
    bool hasDependencyReview(const std::shared_ptr<Flowgraph>& currentFlowgraph) const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_PANE_HH
