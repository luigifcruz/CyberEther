#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_REMOTE_QR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_REMOTE_QR_HH

#include "../context.hh"

#include "../../views/modal/remote_qr.hh"

namespace Jetstream {

struct RemoteQrModalPresenter {
    const PresenterContext& context;

    explicit RemoteQrModalPresenter(const PresenterContext& context) : context(context) {}

    RemoteQrView::Config build() const {
        const auto& remote = context.state.remote;
        return {.inviteUrl = remote.started ? remote.inviteUrl : std::string()};
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_REMOTE_QR_HH
