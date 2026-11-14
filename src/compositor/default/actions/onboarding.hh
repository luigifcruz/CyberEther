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

    explicit OnboardingActions(DefaultCompositorState& state);
    static std::optional<MailOpenOnboarding> Detect(const Settings::Interface& interface, const std::string& version);
    Result handle(const MailOpenOnboarding& msg);
    Result handle(const MailDismissOnboarding&);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_ONBOARDING_HH
