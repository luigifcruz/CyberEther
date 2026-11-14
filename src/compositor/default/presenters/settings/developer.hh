#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_DEVELOPER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_DEVELOPER_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/settings/developer.hh"

namespace Jetstream {

struct DeveloperSettingsPresenter {
    const PresenterContext& context;

    explicit DeveloperSettingsPresenter(const PresenterContext& context);
    DeveloperSettingsView::Config build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_DEVELOPER_HH
