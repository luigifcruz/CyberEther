#include "intelligence_provider.hh"

namespace Jetstream {

IntelligenceProviderPresenter::IntelligenceProviderPresenter(const PresenterContext& context) : context(context) {}

IntelligenceProviderView::Config IntelligenceProviderPresenter::build() const {
    const auto enqueue = context.callbacks.enqueueMail;
    const auto& agent = context.state.agent;
    const auto& test = agent.providerTest;
    const auto& custom = Agent::Provider::Profile::Find(Agent::Provider::Kind::Custom);

    IntelligenceProviderView::Config config;
    for (const auto& profile : Agent::Provider::Profile::All()) {
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
