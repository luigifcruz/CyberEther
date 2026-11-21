#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_GENERAL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_GENERAL_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/settings/general.hh"

#include "render/sakura/themes.hh"

#include <string>

namespace Jetstream {

struct GeneralSettingsPresenter {
    const PresenterContext& context;

    explicit GeneralSettingsPresenter(const PresenterContext& context);
    GeneralSettingsView::Config build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_GENERAL_HH
