#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_BASE_HH

#include "about.hh"
#include "developer.hh"
#include "general.hh"
#include "intelligence.hh"
#include "legal.hh"
#include "maps.hh"
#include "registry.hh"
#include "remote.hh"
#include "runtime.hh"

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/settings/base.hh"

namespace Jetstream {

struct SettingsPresenter {
    const PresenterContext& context;
    GeneralSettingsPresenter general;
    RemoteSettingsPresenter remote;
    MapsSettingsPresenter maps;
    RuntimeSettingsPresenter runtime;
    RegistrySettingsPresenter registry;
    IntelligenceSettingsPresenter intelligence;
    DeveloperSettingsPresenter developer;
    AboutPresenter about;
    LegalSettingsPresenter legal;

    explicit SettingsPresenter(const PresenterContext& context);
    SettingsView::Config build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_BASE_HH
