#include "rename.hh"

namespace Jetstream {

RenameBlockModalPresenter::RenameBlockModalPresenter(const PresenterContext& context) : context(context) {}

std::optional<RenameBlockView::Config> RenameBlockModalPresenter::build() const {
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

}  // namespace Jetstream
