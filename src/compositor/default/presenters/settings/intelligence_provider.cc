#include "intelligence_provider.hh"

namespace Jetstream {

IntelligenceProviderPresenter::IntelligenceProviderPresenter(const PresenterContext& context) : context(context) {}

IntelligenceProviderView::Config IntelligenceProviderPresenter::build() const {
    const auto enqueue = context.callbacks.enqueueMail;
    const auto& agent = context.state.agent;
    const auto& test = agent.providerTest;
    const auto& custom = Agent::Provider::Profile::Find(Agent::Provider::Kind::Custom);

    IntelligenceProviderView::Config config;
    const bool signInAvailable = AgentProviders::SignInAvailable();
    for (const auto& profile : Agent::Provider::Profile::All()) {
        if (profile.kind == Agent::Provider::Kind::ChatGPT && !signInAvailable) {
            continue;
        }
        config.presets.push_back({
            profile.name,
            profile.description,
            AgentProviders::Preset(profile.kind),
        });
    }

    config.defaultPreset = Agent::Provider::Profile::Find(DefaultKind).name;
    config.chatPathHint = custom.chatPath;
    config.modelsPathHint = custom.modelsPath;

    config.session = agent.providers.editSession;
    config.editMode = agent.providers.editTarget.has_value();
    if (config.editMode) {
        config.editProvider = agent.providers.editTarget.value();
        config.editPreset = Agent::Provider::Profile::Find(AgentProviders::Identify(config.editProvider)).name;
    }

    config.saving = agent.providers.keyTask.running;
    config.testRunning = test.running;
    config.testCompleted = test.completed;
    config.testSuccess = test.success;
    config.testRevision = test.revision;
    config.testMessage = test.message;

    const auto& signIn = agent.providers.signIn;
    const bool currentSignIn = signIn.session == agent.providers.editSession;
    config.signInRunning = signIn.running && currentSignIn;
    config.signedIn = currentSignIn && !signIn.record.empty();
    config.signInEmail = config.signedIn ? signIn.email : std::string{};
    config.signInMessage = currentSignIn ? signIn.message : std::string{};
    std::optional<std::string> signInReplace;
    if (agent.providers.editTarget) {
        signInReplace = agent.providers.editTarget->name;
    }
    config.onSignIn = [enqueue, signInReplace](const Settings::Intelligence::Provider& provider, U64 revision) {
        enqueue(MailSignInAgentProvider{.revision = revision, .value = provider, .replaceName = signInReplace});
    };
    config.onCancelSignIn = [enqueue]() {
        enqueue(MailCancelAgentProviderSignIn{});
    };
    config.onManageUsage = [enqueue]() {
        enqueue(MailOpenUrl{.url = AgentChatGpt::UsageUrl, .notifyResult = true});
    };

    for (const auto& provider : AgentProviders(agent).all()) {
        config.existingNames.push_back(provider.name);
    }

    config.isValid = AgentProviders::Valid;
    config.onTest = [enqueue](const Settings::Intelligence::Provider& provider,
                              const std::string& apiKey,
                              U64 revision) {
        enqueue(MailTestAgentProvider{
            .value = provider,
            .revision = revision,
            .apiKey = apiKey,
        });
    };

    std::optional<std::string> replaceName;
    if (config.editMode) {
        replaceName = agent.providers.editTarget->name;
    }
    config.onAdd = [enqueue, replaceName](const Settings::Intelligence::Provider& provider,
                                          const std::string& apiKey) {
        enqueue(MailSaveAgentProvider{
            .value = provider,
            .replaceName = replaceName,
            .apiKey = apiKey,
        });
    };

    return config;
}

}  // namespace Jetstream
