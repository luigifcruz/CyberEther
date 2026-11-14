#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_REMOTE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_REMOTE_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/settings/remote.hh"

namespace Jetstream {

struct RemoteSettingsPresenter {
    const PresenterContext& context;

    explicit RemoteSettingsPresenter(const PresenterContext& context);
    RemoteSettingsView::Config build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_REMOTE_HH
