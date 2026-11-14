#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_RENAME_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_RENAME_HH

#include "../../context.hh"

#include "../../../model/messages.hh"
#include "../../../views/flowgraph/modals/rename.hh"

#include <optional>
#include <string>

namespace Jetstream {

struct RenameBlockModalPresenter {
    const PresenterContext& context;

    explicit RenameBlockModalPresenter(const PresenterContext& context);
    std::optional<RenameBlockView::Config> build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_RENAME_HH
