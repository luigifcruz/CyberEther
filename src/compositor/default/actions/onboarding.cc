#include "onboarding.hh"

namespace Jetstream {

OnboardingActions::OnboardingActions(DefaultCompositorState& state) : state(state) {}

std::optional<MailOpenOnboarding> OnboardingActions::Detect(const Settings::Interface& interface, const std::string& version) {
    if (interface.welcomeVersion == version) {
        return std::nullopt;
    }
    const bool fresh = interface.welcomeVersion.empty() &&
                       interface.recentFlowgraphs.empty() &&
                       interface.favoriteFlowgraphs.empty();
    return MailOpenOnboarding{.page = fresh ? OnboardingPage::Welcome : OnboardingPage::WhatsNew};
}

Result OnboardingActions::handle(const MailOpenOnboarding& msg) {
    state.modal = {.content = ModalContent::Onboarding, .onboardingPage = msg.page};
    return Result::SUCCESS;
}

Result OnboardingActions::handle(const MailDismissOnboarding&) {
    if (state.modal.content == ModalContent::Onboarding) {
        state.modal = {};
    }
    JST_CHECK(PersistSettings([](Settings& settings) {
        settings.interface.welcomeVersion = JETSTREAM_VERSION_STR;
    }));
    return Result::SUCCESS;
}

}  // namespace Jetstream
