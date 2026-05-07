#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_LEGAL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_LEGAL_HH

#include "../../context.hh"

#include "../../../model/messages.hh"
#include "../../../views/modal/settings/legal.hh"

namespace Jetstream {

struct LegalSettingsPresenter {
    const PresenterContext& context;

    explicit LegalSettingsPresenter(const PresenterContext& context) : context(context) {}

    LegalSettingsPanel::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        return LegalSettingsPanel::Config{
            .onViewFullLicenses = [enqueue]() {
                enqueue(MailOpenUrl{.url = "https://cyberether.org/docs/acknowledgments"});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_LEGAL_HH
