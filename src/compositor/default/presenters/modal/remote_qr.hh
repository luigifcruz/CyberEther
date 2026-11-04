#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_REMOTE_QR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_REMOTE_QR_HH

#include "../context.hh"
#include "../../model/messages.hh"

#include "../../views/modal/remote_qr.hh"

#include <string>

namespace Jetstream {

struct RemoteQrModalPresenter {
    const PresenterContext& context;

    explicit RemoteQrModalPresenter(const PresenterContext& context) : context(context) {}

    RemoteQrView::Config build() const {
        const auto& remote = context.state.remote;
        const auto enqueue = context.callbacks.enqueueMail;
        const std::string url = remote.started ? remote.inviteUrl : std::string();
        return {
            .inviteUrl = url,
            .onCopyLink = [enqueue, url]() {
                enqueue(MailCopyText{.label = "Invite link", .value = url});
            },
            .onClose = [enqueue]() {
                enqueue(MailCloseModal{});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_REMOTE_QR_HH
