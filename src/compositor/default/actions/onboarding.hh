#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_ONBOARDING_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_ONBOARDING_HH

#include "persist.hh"
#include "../model/messages.hh"
#include "../model/state.hh"

#include "jetstream/config.hh"
#include "jetstream/settings.hh"

#include <optional>
#include <string>
#include <tuple>

namespace Jetstream {

struct OnboardingActions {
    using Filter = std::tuple<MailOpenOnboarding,
                              MailDismissOnboarding>;

    DefaultCompositorState& state;

    explicit OnboardingActions(DefaultCompositorState& state) : state(state) {}

    static std::optional<MailOpenOnboarding> Detect(const Settings::Interface& interface, const std::string& version) {
        if (interface.welcomeVersion == version) {
            return std::nullopt;
        }
        const bool fresh = interface.welcomeVersion.empty() &&
                           interface.recentFlowgraphs.empty() &&
                           interface.favoriteFlowgraphs.empty();
        if (fresh) {
            return std::nullopt;
        }
        return MailOpenOnboarding{};
    }

    Result handle(const MailOpenOnboarding&) {
        state.modal = {.content = ModalContent::Onboarding};
        return Result::SUCCESS;
    }

    Result handle(const MailDismissOnboarding&) {
        if (state.modal.content == ModalContent::Onboarding) {
            state.modal = {};
        }
        JST_CHECK(PersistSettings([](Settings& settings) {
            settings.interface.welcomeVersion = JETSTREAM_VERSION_STR;
        }));
        return Result::SUCCESS;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_ONBOARDING_HH
