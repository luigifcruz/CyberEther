#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_MAPS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_MAPS_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/settings/maps.hh"

namespace Jetstream {

struct MapsSettingsPresenter {
    const PresenterContext& context;

    explicit MapsSettingsPresenter(const PresenterContext& context);
    MapsSettingsView::Config build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_MAPS_HH
