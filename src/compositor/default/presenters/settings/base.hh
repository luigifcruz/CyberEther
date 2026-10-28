#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_BASE_HH

#include "about.hh"
#include "developer.hh"
#include "general.hh"
#include "intelligence.hh"
#include "legal.hh"
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
    RuntimeSettingsPresenter runtime;
    RegistrySettingsPresenter registry;
    IntelligenceSettingsPresenter intelligence;
    DeveloperSettingsPresenter developer;
    AboutPresenter about;
    LegalSettingsPresenter legal;

    explicit SettingsPresenter(const PresenterContext& context) : context(context),
                                                                  general(context),
                                                                  remote(context),
                                                                  runtime(context),
                                                                  registry(context),
                                                                  intelligence(context),
                                                                  developer(context),
                                                                  about(context),
                                                                  legal(context) {}

    SettingsView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        SettingsView::Config config;
        config.section = context.state.settings.section;
        switch (config.section) {
            case SettingsSection::General:
                config.general = general.build();
                break;
            case SettingsSection::Remote:
                config.remote = remote.build();
                break;
            case SettingsSection::Runtime:
                config.runtime = runtime.build();
                break;
            case SettingsSection::Registry:
                config.registry = registry.build();
                break;
            case SettingsSection::Intelligence:
                config.intelligence = intelligence.build();
                break;
            case SettingsSection::Developer:
                config.developer = developer.build();
                break;
            case SettingsSection::About:
                config.about = about.build();
                break;
            case SettingsSection::Legal:
                config.legal = legal.build();
                break;
        }
        config.onSectionChange = [enqueue](SettingsSection section) {
            enqueue(MailSetSettingsSection{.section = section});
        };
        config.onClose = [enqueue]() {
            enqueue(MailCloseSettings{});
        };
        config.onFocus = [enqueue]() {
            enqueue(MailFocusSettings{});
        };
        return config;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_BASE_HH
