#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_BASE_HH

#include "node.hh"
#include "picker.hh"

#include "../../context.hh"

#include "../../../model/messages.hh"
#include "../../../model/meta.hh"
#include "../../../views/flowgraph/editor/base.hh"

#include "jetstream/block.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_view.hh"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphEditorPresenter {
    const PresenterContext& context;
    FlowgraphCatalogPresenter catalog;
    FlowgraphNodePresenter node;

    explicit FlowgraphEditorPresenter(const PresenterContext& context);
    FlowgraphEditor::Config build(const std::string& flowgraphId,
                                  const std::shared_ptr<Flowgraph>& flowgraph) const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_EDITOR_BASE_HH
