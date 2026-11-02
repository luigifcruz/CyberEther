#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_START_TAB_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_START_TAB_BASE_HH

#include "welcome.hh"

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/start_tab/base.hh"

#include <optional>

namespace Jetstream {

struct StartTabPresenter {
    const PresenterContext& context;
    WelcomePresenter welcome;

    explicit StartTabPresenter(const PresenterContext& context) : context(context), welcome(context) {}

    std::optional<StartTabView::Config> build() const {
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
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_START_TAB_BASE_HH
