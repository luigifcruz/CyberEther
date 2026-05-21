#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_BASE_HH

#include "about.hh"
#include "developer.hh"
#include "general.hh"
#include "legal.hh"
#include "registry.hh"
#include "remote.hh"

#include "../../context.hh"

#include "../../../model/messages.hh"
#include "../../../views/modal/settings/base.hh"

namespace Jetstream {

struct SettingsModalPresenter {
    const PresenterContext& context;
    GeneralSettingsPresenter general;
    RemoteSettingsPresenter remote;
    RegistrySettingsPresenter registry;
    DeveloperSettingsPresenter developer;
    AboutPresenter about;
    LegalSettingsPresenter legal;

    explicit SettingsModalPresenter(const PresenterContext& context) : context(context),
                                                                       general(context),
                                                                       remote(context),
                                                                       registry(context),
                                                                       developer(context),
                                                                       about(context),
                                                                       legal(context) {}

    SettingsView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        SettingsView::Config config;
        config.section = context.state.settings.section;
        config.general = general.build();
        config.remote = remote.build();
        config.registry = registry.build();
        config.developer = developer.build();
        config.about = about.build();
        config.legal = legal.build();
        config.onSectionChange = [enqueue](SettingsSection section) {
            enqueue(MailSetSettingsSection{.section = section});
        };
        return config;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_BASE_HH
