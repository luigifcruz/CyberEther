#include "base.hh"

namespace Jetstream {

SettingsPresenter::SettingsPresenter(const PresenterContext& context) : context(context),
                                                                        general(context),
                                                                        remote(context),
                                                                        maps(context),
                                                                        runtime(context),
                                                                        registry(context),
                                                                        intelligence(context),
                                                                        developer(context),
                                                                        about(context),
                                                                        legal(context) {}

SettingsView::Config SettingsPresenter::build() const {
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
        case SettingsSection::Maps:
            config.maps = maps.build();
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

}  // namespace Jetstream
