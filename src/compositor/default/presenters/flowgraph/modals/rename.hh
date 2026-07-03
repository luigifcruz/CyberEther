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
        const std::optional<std::string> targetFlowgraph = context.state.modal.flowgraph.has_value()
            ? context.state.modal.flowgraph
            : context.state.interface.focusedFlowgraph;
        if (!targetFlowgraph.has_value() || !context.state.modal.renameBlockOldName.has_value()) {
            return std::nullopt;
        }

        const auto enqueue = context.callbacks.enqueueMail;
        const std::string focusedFlowgraph = targetFlowgraph.value();
        const std::string oldName = context.state.modal.renameBlockOldName.value();
        return RenameBlockView::Config{
            .oldName = oldName,
            .onRename = [enqueue, focusedFlowgraph, oldName](const std::string& newName) {
                enqueue(MailRenameBlock{focusedFlowgraph, oldName, newName});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_MODALS_RENAME_HH
