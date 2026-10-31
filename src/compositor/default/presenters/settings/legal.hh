#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_LEGAL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_LEGAL_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/settings/legal.hh"

namespace Jetstream {

struct LegalSettingsPresenter {
    const PresenterContext& context;

    explicit LegalSettingsPresenter(const PresenterContext& context) : context(context) {}

    LegalSettingsView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        return LegalSettingsView::Config{
            .onViewFullLicenses = [enqueue]() {
                enqueue(MailOpenUrl{.url = "https://cyberether.org/docs/acknowledgments"});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_LEGAL_HH
