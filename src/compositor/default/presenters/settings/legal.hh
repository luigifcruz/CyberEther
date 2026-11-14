#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_LEGAL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_LEGAL_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/settings/legal.hh"

namespace Jetstream {

struct LegalSettingsPresenter {
    const PresenterContext& context;

    explicit LegalSettingsPresenter(const PresenterContext& context);
    LegalSettingsView::Config build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_LEGAL_HH
