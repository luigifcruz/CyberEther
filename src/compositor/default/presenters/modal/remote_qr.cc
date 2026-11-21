#include "remote_qr.hh"

namespace Jetstream {

RemoteQrModalPresenter::RemoteQrModalPresenter(const PresenterContext& context) : context(context) {}

RemoteQrView::Config RemoteQrModalPresenter::build() const {
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

}  // namespace Jetstream
