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

    explicit RenameBlockModalPresenter(const PresenterContext& context) : context(context) {}

    std::optional<RenameBlockView::Config> build() const {
        if (!context.state.interface.focusedFlowgraph.has_value() ||
            !context.state.modal.renameBlockOldName.has_value()) {
            return std::nullopt;
        }

        const auto enqueue = context.callbacks.enqueueMail;
        const std::string focusedFlowgraph = context.state.interface.focusedFlowgraph.value();
        const std::string oldName = context.state.modal.renameBlockOldName.value();
        return RenameBlockView::Config{
            .oldName = oldName,
            .onRename = [enqueue, focusedFlowgraph, oldName](const std::string& newName) {
                enqueue(MailRenameBlock{focusedFlowgraph, oldName, newName});
                enqueue(MailCloseModal{});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_RENAME_HH
