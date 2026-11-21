#include "base.hh"

namespace Jetstream {

StartTabPresenter::StartTabPresenter(const PresenterContext& context) : context(context), welcome(context) {}

std::optional<StartTabView::Config> StartTabPresenter::build() const {
    if (!context.state.startTabShown()) {
        return std::nullopt;
    }
    const auto enqueue = context.callbacks.enqueueMail;
    StartTabView::Config config{
        .welcome = welcome.build(),
        .closable = context.state.workspaceTabs() > 0,
        .onClose = [enqueue]() {
            enqueue(MailCloseStartTab{});
        },
        .onFocus = [enqueue]() {
            enqueue(MailFocusStartTab{});
        },
    };
    return config;
}

}  // namespace Jetstream
