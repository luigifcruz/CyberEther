#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_INTELLIGENCE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_INTELLIGENCE_HH

#include "../../context.hh"
#include "../../agent/labels.hh"
#include "../../../model/messages.hh"
#include "../../../views/modal/settings/intelligence.hh"

#include <map>
#include <string>

namespace Jetstream {

struct IntelligenceSettingsPresenter {
    const PresenterContext& context;

    explicit IntelligenceSettingsPresenter(const PresenterContext& context) : context(context) {}

    IntelligenceSettingsPanel::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        const auto& agent = context.state.agent;

        IntelligenceSettingsPanel::Config config;
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
                .context = AgentLabels::Context(model.contextLength),
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
        voiceRow.models.assign(models.begin(), models.end());
        voiceRow.model = voice.model;

        const auto voices = Agent::Live::Voices();
        voiceRow.voices.assign(voices.begin(), voices.end());
        voiceRow.voice = voice.voice;

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
        config.onVoiceModelChange = [enqueue](const std::string& model) {
            enqueue(MailSetAgentVoiceModel{.value = model});
        };
        config.onVoiceChange = [enqueue](const std::string& name) {
            enqueue(MailSetAgentVoiceName{.value = name});
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

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_SETTINGS_INTELLIGENCE_HH
