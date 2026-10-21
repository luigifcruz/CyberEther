#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_INTELLIGENCE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_INTELLIGENCE_HH

#include "../context.hh"
#include "../agent/labels.hh"
#include "intelligence_model.hh"
#include "intelligence_provider.hh"
#include "../../model/messages.hh"
#include "../../views/settings/intelligence.hh"

#include <map>
#include <string>
#include <vector>

namespace Jetstream {

struct IntelligenceSettingsPresenter {
    const PresenterContext& context;

    explicit IntelligenceSettingsPresenter(const PresenterContext& context) : context(context) {}

    static std::vector<std::string> modelTags(const Settings::Intelligence::Model& model) {
        std::vector<std::string> tags;
        if (model.contextLength > 0) {
            tags.push_back(AgentLabels::Tokens(model.contextLength) + " context");
        }
        if (model.priced) {
            tags.push_back(AgentLabels::Price(model.inputCost) + " in");
            tags.push_back(AgentLabels::Price(model.outputCost) + " out");
        }
        return tags;
    }

    IntelligenceSettingsPanel::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        const auto& agent = context.state.agent;

        IntelligenceSettingsPanel::Config config;
        config.page = context.state.settings.intelligencePage;
        if (config.page == IntelligencePage::Provider) {
            config.provider = IntelligenceProviderPresenter(context).build();
        } else if (config.page == IntelligencePage::AddModel) {
            config.model = IntelligenceModelPresenter(context).build();
        }
        config.enabled = agent.settings.enabled;
        config.onEnabledChange = [enqueue](bool value) {
            enqueue(MailSetAgentEnabled{.value = value});
        };

        const AgentProviders providers(agent);

        std::map<std::string, AgentProviderStatus> statuses;
        for (const auto& provider : providers.all()) {
            const auto status = AgentLabels::Status(agent, provider);
            statuses.try_emplace(provider.name, status);
            config.providers.push_back({
                .name = provider.name,
                .status = status.label,
                .ready = status.ready,
                .pending = status.pending,
            });
        }

        for (const auto& model : agent.settings.models) {
            config.models.push_back({
                .provider = model.provider,
                .id = model.id,
                .label = AgentLabels::Model(model),
                .tags = modelTags(model),
            });
        }

        const auto availability = providers.availability();
        const auto& voice = availability.voice;
        const auto backends = providers.backends();

        auto& voiceRow = config.voice;
        voiceRow.available = availability.state != AgentProviders::VoiceAvailability::State::Unsupported;
        for (U64 i = 0; i < backends.size(); ++i) {
            const auto& backend = backends[i];
            const auto label = AgentLabels::Model(backend);
            voiceRow.backends.push_back(jst::fmt::format("{} ({})", label, backend.provider));
            if (AgentProviders::Same(backend, voice.backend)) {
                voiceRow.backend = i;
            }
        }

        const auto models = Agent::Live::Models();
        for (const auto& model : models) {
            voiceRow.models.push_back(AgentLabels::Title(model));
        }
        voiceRow.model = AgentLabels::Title(voice.model);

        const auto voices = Agent::Live::Voices();
        for (const auto& name : voices) {
            voiceRow.voices.push_back(AgentLabels::Title(name));
        }
        voiceRow.voice = AgentLabels::Title(voice.voice);

        if (availability.state == AgentProviders::VoiceAvailability::State::Ready) {
            const auto cached = statuses.find(voice.backend.provider);
            AgentProviderStatus status;
            if (cached != statuses.end()) {
                status = cached->second;
            } else {
                const auto provider = providers.find(voice.backend.provider);
                status = AgentLabels::Status(agent, provider);
            }
            voiceRow.ready = status.ready;
            voiceRow.pending = status.pending;
            voiceRow.status = std::move(status.label);
        } else {
            voiceRow.status = AgentLabels::Voice(availability.state);
        }

        config.onVoiceBackendSelect = [enqueue, backends](U64 index) {
            if (index < backends.size()) {
                enqueue(MailSetAgentVoiceBackend{.value = backends[index]});
            }
        };
        config.onVoiceModelChange = [enqueue, models](const std::string& label) {
            for (const auto& model : models) {
                if (AgentLabels::Title(model) == label) {
                    enqueue(MailSetAgentVoiceModel{.value = model});
                    return;
                }
            }
        };
        config.onVoiceChange = [enqueue, voices](const std::string& label) {
            for (const auto& name : voices) {
                if (AgentLabels::Title(name) == label) {
                    enqueue(MailSetAgentVoiceName{.value = name});
                    return;
                }
            }
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
        config.onRemoveModel = [enqueue](const IntelligenceSettingsPanel::ModelRow& model) {
            enqueue(MailRemoveAgentModel{.value = {model.provider, model.id}});
        };

        return config;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_INTELLIGENCE_HH
