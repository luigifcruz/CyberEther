#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_REMOTE_QR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_REMOTE_QR_HH

#include "../context.hh"
#include "../../model/messages.hh"

#include "../../views/modal/remote_qr.hh"

#include <string>

namespace Jetstream {

struct RemoteQrModalPresenter {
    const PresenterContext& context;

    explicit RemoteQrModalPresenter(const PresenterContext& context);
    RemoteQrView::Config build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_REMOTE_QR_HH
