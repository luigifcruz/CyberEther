#include "legal.hh"

namespace Jetstream {

LegalSettingsPresenter::LegalSettingsPresenter(const PresenterContext& context) : context(context) {}

LegalSettingsView::Config LegalSettingsPresenter::build() const {
    const auto enqueue = context.callbacks.enqueueMail;
    return LegalSettingsView::Config{
        .onViewFullLicenses = [enqueue]() {
            enqueue(MailOpenUrl{.url = "https://cyberether.org/docs/acknowledgments"});
        },
    };
}

}  // namespace Jetstream
