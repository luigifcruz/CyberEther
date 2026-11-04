#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/onboarding.hh"
#include "environment.hh"

#include "jetstream/config.hh"

#include <string>

using namespace Jetstream;

namespace {

Settings::Interface::FlowgraphEntry Entry(const std::string& path) {
    return {.path = path, .title = "Radio", .accessed = 1};
}

}  // namespace

TEST_CASE("Onboarding stays hidden when the welcome version matches the running build",
          "[core][compositor][onboarding]") {
    Settings::Interface interface;
    interface.welcomeVersion = JETSTREAM_VERSION_STR;
    CHECK_FALSE(OnboardingActions::Detect(interface, JETSTREAM_VERSION_STR).has_value());

    interface.recentFlowgraphs = {Entry("/flows/radio.yml")};
    CHECK_FALSE(OnboardingActions::Detect(interface, JETSTREAM_VERSION_STR).has_value());
}

TEST_CASE("Onboarding stays hidden on a first install", "[core][compositor][onboarding]") {
    const Settings::Interface interface;
    CHECK_FALSE(OnboardingActions::Detect(interface, "2.0.0").has_value());
}

TEST_CASE("Onboarding opens after an update", "[core][compositor][onboarding]") {
    Settings::Interface interface;
    interface.welcomeVersion = "1.19.0";
    auto mail = OnboardingActions::Detect(interface, "2.0.0");
    REQUIRE(mail.has_value());

    interface.welcomeVersion = "3.0.0";
    mail = OnboardingActions::Detect(interface, "2.0.0");
    REQUIRE(mail.has_value());
}

TEST_CASE("Onboarding treats an empty welcome version with saved flowgraphs as an update",
          "[core][compositor][onboarding]") {
    Settings::Interface recents;
    recents.recentFlowgraphs = {Entry("/flows/radio.yml")};
    auto mail = OnboardingActions::Detect(recents, "2.0.0");
    REQUIRE(mail.has_value());

    Settings::Interface favorites;
    favorites.favoriteFlowgraphs = {Entry("/flows/fm.yml")};
    mail = OnboardingActions::Detect(favorites, "2.0.0");
    REQUIRE(mail.has_value());
}

TEST_CASE("Opening onboarding shows the modal", "[core][compositor][onboarding]") {
    DefaultCompositorState state;
    OnboardingActions actions(state);
    state.modal.flowgraph = "a";

    REQUIRE(actions.handle(MailOpenOnboarding{}) == Result::SUCCESS);
    CHECK(state.modal.content == std::optional<ModalContent>(ModalContent::Onboarding));
    CHECK_FALSE(state.modal.flowgraph.has_value());
}

TEST_CASE("Dismissing onboarding closes it and persists the running version",
          "[core][compositor][onboarding]") {
    TestSupport::SettingsSandbox sandbox("compositor-onboarding");
    sandbox.require();
    DefaultCompositorState state;
    OnboardingActions actions(state);

    REQUIRE(actions.handle(MailOpenOnboarding{}) == Result::SUCCESS);
    REQUIRE(actions.handle(MailDismissOnboarding{}) == Result::SUCCESS);
    CHECK_FALSE(state.modal.content.has_value());

    REQUIRE(Settings::Flush() == Result::SUCCESS);
    Settings settings;
    REQUIRE(Settings::Get(settings) == Result::SUCCESS);
    CHECK(settings.interface.welcomeVersion == JETSTREAM_VERSION_STR);
    CHECK_FALSE(OnboardingActions::Detect(settings.interface, JETSTREAM_VERSION_STR).has_value());
}

TEST_CASE("Dismissing onboarding leaves a modal that replaced it open", "[core][compositor][onboarding]") {
    TestSupport::SettingsSandbox sandbox("compositor-onboarding-replaced");
    sandbox.require();
    DefaultCompositorState state;
    OnboardingActions actions(state);
    state.modal.content = ModalContent::Feedback;

    REQUIRE(actions.handle(MailDismissOnboarding{}) == Result::SUCCESS);
    CHECK(state.modal.content == std::optional<ModalContent>(ModalContent::Feedback));
}
