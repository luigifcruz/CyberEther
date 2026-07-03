#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_HUD_REMOTE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_HUD_REMOTE_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/hud/remote.hh"

#include <optional>

namespace Jetstream {

struct RemoteHudPresenter {
    const PresenterContext& context;

    explicit RemoteHudPresenter(const PresenterContext& context) : context(context) {}

    std::optional<RemoteHudView::Config> build() const {
        if (!context.state.remote.started) {
            return std::nullopt;
        }

        const auto enqueue = context.callbacks.enqueueMail;
        return RemoteHudView::Config{
            .clientCount = context.state.remote.clientCount,
            .onOpen = [enqueue]() {
                enqueue(MailOpenModal{.content = ModalContent::RemoteStreaming});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_HUD_REMOTE_HH
