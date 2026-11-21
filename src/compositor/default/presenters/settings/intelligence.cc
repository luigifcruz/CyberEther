#include "intelligence.hh"

namespace Jetstream {

IntelligenceSettingsPresenter::IntelligenceSettingsPresenter(const PresenterContext& context) : context(context),
                                                                                                provider(context),
                                                                                                model(context) {}

std::vector<std::string> IntelligenceSettingsPresenter::ModelTags(const Settings::Intelligence::Model& model) {
    std::vector<std::string> tags;
    if (model.contextLength > 0) {
        tags.push_back(jst::fmt::format("{} context", AgentLabels::Tokens(model.contextLength)));
    }
    if (model.priced) {
        tags.push_back(jst::fmt::format("{} in", AgentLabels::Price(model.inputCost)));
        tags.push_back(jst::fmt::format("{} out", AgentLabels::Price(model.outputCost)));
    }
    return tags;
}

StatusBadge::Config IntelligenceSettingsPresenter::Badge(const AgentProviderStatus& status) {
    if (status.ready) {
        return {.label = status.label, .tone = StatusBadge::Tone::Success};
    }
    return {
        .label = status.label,
        .tone = status.pending ? StatusBadge::Tone::Neutral : StatusBadge::Tone::Warning,
    };
}

IntelligenceSettingsView::Config IntelligenceSettingsPresenter::build() const {
    const auto enqueue = context.callbacks.enqueueMail;
    const auto& agent = context.state.agent;

    IntelligenceSettingsView::Config config;
    config.page = context.state.settings.intelligencePage;
    if (config.page == IntelligencePage::Provider) {
        config.provider = provider.build();
    } else if (config.page == IntelligencePage::AddModel) {
        config.model = model.build();
    }
    config.enabled = agent.settings.enabled;
    config.onNavigate = [enqueue](IntelligencePage page) {
        enqueue(MailSetIntelligencePage{.page = page});
    };
    config.onEnabledChange = [enqueue](bool value) {
        enqueue(MailSetAgentEnabled{.value = value});
    };

    const AgentProviders providers(agent);

    std::map<std::string, AgentProviderStatus> statuses;
    for (const auto& entry : providers.all()) {
        const auto status = AgentLabels::Status(agent, entry);
        statuses.try_emplace(entry.name, status);
        config.providers.push_back({.name = entry.name, .status = Badge(status)});
    }

    for (const auto& entry : agent.settings.models) {
        config.models.push_back({
            .provider = entry.provider,
            .id = entry.id,
            .label = AgentLabels::Model(entry),
            .tags = ModelTags(entry),
        });
    }

    const auto availability = providers.availability();
    const auto& voice = availability.voice;
    const auto backends = providers.backends();

    auto& voiceRow = config.voice;
    voiceRow.available = availability.state != AgentProviders::VoiceAvailability::State::Unsupported;
    for (U64 i = 0; i < backends.size(); ++i) {
        const auto& backend = backends[i];
        voiceRow.backends.push_back(jst::fmt::format("{} ({})", AgentLabels::Model(backend), backend.provider));
        if (AgentProviders::Same(backend, voice.backend)) {
            voiceRow.backend = i;
        }
    }

    const auto models = Agent::Live::Models();
    auto modelOptions = TitledOptions(models, voice.model);
    voiceRow.models = std::move(modelOptions.labels);
    voiceRow.model = modelOptions.selected;
    const auto voices = Agent::Live::Voices();
    auto voiceOptions = TitledOptions(voices, voice.voice);
    voiceRow.voices = std::move(voiceOptions.labels);
    voiceRow.voice = voiceOptions.selected;

    if (availability.state == AgentProviders::VoiceAvailability::State::Ready) {
        const auto cached = statuses.find(voice.backend.provider);
        voiceRow.status = Badge(cached != statuses.end()
                                    ? cached->second
                                    : AgentLabels::Status(agent, providers.find(voice.backend.provider)));
    } else {
        voiceRow.status = {
            .label = AgentLabels::Voice(availability.state),
            .tone = voiceRow.available ? StatusBadge::Tone::Warning : StatusBadge::Tone::Neutral,
        };
    }

    config.onVoiceBackendSelect = [enqueue, backends](U64 index) {
        if (index < backends.size()) {
            enqueue(MailSetAgentVoiceBackend{.value = backends[index]});
        }
    };
    config.onVoiceModelSelect = [enqueue, models](U64 index) {
        if (index < models.size()) {
            enqueue(MailSetAgentVoiceModel{.value = models[index]});
        }
    };
    config.onVoiceSelect = [enqueue, voices](U64 index) {
        if (index < voices.size()) {
            enqueue(MailSetAgentVoiceName{.value = voices[index]});
        }
    };

    config.mcp = {
        .settings = agent.settings.mcp,
        .url = agent.mcp.url,
        .error = agent.mcp.error,
    };
    config.onMcpChange = [enqueue](const Settings::Intelligence::Mcp& value) {
        enqueue(MailSetAgentMcp{.value = value});
    };

    config.onOpenAddProvider = [enqueue]() {
        enqueue(MailOpenAgentProviderEditor{});
    };
    config.onEditProvider = [enqueue](const std::string& name) {
        enqueue(MailOpenAgentProviderEditor{.name = name});
    };
    config.onRemoveProvider = [enqueue](const std::string& name) {
        enqueue(MailRemoveAgentProvider{.name = name});
    };
    config.onOpenAddModel = [enqueue]() {
        enqueue(MailOpenAgentModelPicker{});
    };
    config.onRemoveModel = [enqueue](const IntelligenceSettingsView::ModelRow& model) {
        enqueue(MailRemoveAgentModel{.value = {model.provider, model.id}});
    };

    return config;
}

}  // namespace Jetstream
