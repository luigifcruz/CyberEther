#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_HUD_REMOTE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_HUD_REMOTE_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/hud/remote.hh"

namespace Jetstream {

struct RemoteHudPresenter {
    const PresenterContext& context;

    explicit RemoteHudPresenter(const PresenterContext& context) : context(context) {}

    RemoteHudView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        return RemoteHudView::Config{
            .visible = context.state.remote.started,
            .clientCount = context.state.remote.started ? context.state.remote.clientCount : 0,
            .onOpen = [enqueue]() {
                enqueue(MailOpenModal{.content = ModalContent::RemoteStreaming});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_HUD_REMOTE_HH
