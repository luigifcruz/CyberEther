#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_INTELLIGENCE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_INTELLIGENCE_HH

#include "../components/compact_button.hh"
#include "../components/section_group.hh"
#include "../components/status_badge.hh"
#include "intelligence_model.hh"
#include "intelligence_provider.hh"
#include "components/settings_header.hh"

#include "../../model/ui.hh"

#include "jetstream/render/sakura/base.hh"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct IntelligenceSettingsView {
    struct ProviderRow {
        std::string name;
        StatusBadge::Config status;
    };

    struct ModelRow {
        std::string provider;
        std::string id;
        std::string label;
        std::vector<std::string> tags;
    };

    struct VoiceRow {
        StatusBadge::Config status;
        bool available = false;
        std::vector<std::string> backends;
        std::optional<U64> backend;
        std::vector<std::string> models;
        std::optional<U64> model;
        std::vector<std::string> voices;
        std::optional<U64> voice;
    };

    struct Config {
        IntelligencePage page = IntelligencePage::Overview;
        IntelligenceProviderView::Config provider;
        IntelligenceModelView::Config model;
        bool enabled = true;
        std::vector<ProviderRow> providers;
        std::vector<ModelRow> models;
        VoiceRow voice;
        std::function<void(IntelligencePage)> onNavigate;
        std::function<void(bool)> onEnabledChange;
        std::function<void(U64)> onVoiceBackendSelect;
        std::function<void(U64)> onVoiceModelSelect;
        std::function<void(U64)> onVoiceSelect;
        std::function<void()> onOpenAddProvider;
        std::function<void(const std::string&)> onEditProvider;
        std::function<void(const std::string&)> onRemoveProvider;
        std::function<void()> onOpenAddModel;
        std::function<void(const ModelRow&)> onRemoveModel;
    };

    void update(Config config) {
        this->config = std::move(config);

        if (this->config.enabled && this->config.page == IntelligencePage::Provider) {
            this->config.provider.onBack = back();
            providerPage.update(this->config.provider);
            stack.update({.items = {{.child = &providerPage.content()}}});
            return;
        }
        if (this->config.enabled && this->config.page == IntelligencePage::AddModel) {
            this->config.model.onBack = back();
            modelPage.update(this->config.model);
            stack.update({.items = {{.child = &modelPage.content()}}});
            return;
        }

        header.update({
            .title = "Intelligence",
            .description = "Configure intelligence features and model access.",
        });

        enabledToggle.update({
            .value = this->config.enabled,
            .onChange = [this](bool value) {
                if (this->config.onEnabledChange) {
                    this->config.onEnabledChange(value);
                }
            },
        });
        assistantGroup.update({
            .title = "Assistant",
            .description = "Chat and voice features powered by your model providers.",
            .rows = {{
                .label = "Enable Intelligence",
                .description = "When off, all AI features are hidden and providers are disconnected.",
                .control = &enabledToggle,
            }},
        });

        std::vector<Sakura::Stack::Item> items = {
            {.child = &header},
            {.child = &assistantGroup},
        };
        if (this->config.enabled) {
            updateProviders();
            updateModels();
            updateVoice();
            items.push_back({.child = &providersGroup});
            items.push_back({.child = &modelsGroup});
            items.push_back({.child = &voiceGroup});
        }
        stack.update({
            .spacing = 4.0f,
            .items = std::move(items),
        });
    }

    Sakura::Component& content() {
        return stack;
    }

 private:
    struct ProviderControls {
        StatusBadge status;
        CompactButton edit;
        CompactButton remove;
        Sakura::Stack row;
    };

    struct ModelControls {
        std::vector<std::unique_ptr<StatusBadge>> tags;
        CompactButton remove;
        Sakura::Stack row;
    };

    std::function<void()> back() {
        return [this]() {
            if (config.onNavigate) {
                config.onNavigate(IntelligencePage::Overview);
            }
        };
    }

    void updateProviders() {
        while (providerControls.size() < config.providers.size()) {
            providerControls.push_back(std::make_unique<ProviderControls>());
        }
        std::vector<Sakura::SettingCard::Row> rows;
        rows.reserve(config.providers.size());
        for (U64 index = 0; index < config.providers.size(); ++index) {
            const auto& provider = config.providers[index];
            auto& controls = *providerControls[index];
            controls.status.update(provider.status);
            controls.edit.update({
                .str = "Edit",
                .onClick = [this, name = provider.name]() {
                    if (config.onEditProvider) {
                        config.onEditProvider(name);
                    }
                },
            });
            controls.remove.update({
                .str = "Remove",
                .onClick = [this, name = provider.name]() {
                    if (config.onRemoveProvider) {
                        config.onRemoveProvider(name);
                    }
                },
            });
            controls.row.update({
                .axis = Sakura::Stack::Axis::Horizontal,
                .spacing = 6.0f,
                .items = {
                    {.child = &controls.status},
                    {.child = &controls.edit},
                    {.child = &controls.remove},
                },
            });
            rows.push_back({.label = provider.name, .control = &controls.row});
        }
        addProviderButton.update({
            .str = "Add",
            .tone = CompactButton::Tone::Primary,
            .onClick = [this]() {
                if (config.onOpenAddProvider) {
                    config.onOpenAddProvider();
                }
            },
        });
        providersGroup.update({
            .title = "Providers",
            .description = "OpenAI-compatible endpoints the agent can use.",
            .rows = std::move(rows),
            .empty = "No providers configured. Add one to enable the agent.",
            .action = &addProviderButton,
        });
    }

    void updateModels() {
        while (modelControls.size() < config.models.size()) {
            modelControls.push_back(std::make_unique<ModelControls>());
        }
        std::vector<Sakura::SettingCard::Row> rows;
        rows.reserve(config.models.size());
        for (U64 index = 0; index < config.models.size(); ++index) {
            const auto& model = config.models[index];
            auto& controls = *modelControls[index];
            controls.remove.update({
                .str = "Remove",
                .onClick = [this, model]() {
                    if (config.onRemoveModel) {
                        config.onRemoveModel(model);
                    }
                },
            });
            while (controls.tags.size() < model.tags.size()) {
                controls.tags.push_back(std::make_unique<StatusBadge>());
            }
            std::vector<Sakura::Stack::Item> items;
            for (U64 tag = 0; tag < model.tags.size(); ++tag) {
                controls.tags[tag]->update({.label = model.tags[tag]});
                items.push_back({.child = controls.tags[tag].get(), .shrink = 1.0f});
            }
            items.push_back({.child = &controls.remove});
            controls.row.update({
                .axis = Sakura::Stack::Axis::Horizontal,
                .spacing = 6.0f,
                .items = std::move(items),
            });
            rows.push_back({
                .label = model.label,
                .description = model.provider,
                .control = &controls.row,
                .wide = true,
            });
        }
        addModelButton.update({
            .str = "Add",
            .tone = CompactButton::Tone::Primary,
            .onClick = [this]() {
                if (config.onOpenAddModel) {
                    config.onOpenAddModel();
                }
            },
        });
        modelsGroup.update({
            .title = "Models",
            .description = "Models available to the agent chat.",
            .rows = std::move(rows),
            .empty = "No models configured. Add one to enable the agent.",
            .action = &addModelButton,
        });
    }

    void updateVoice() {
        const auto& voice = config.voice;
        const bool voiceEnabled = voice.available && !voice.backends.empty();

        voiceBackendCombo.update(selection("settings:intelligence-voice-backend", voice.backends, voice.backend,
                                           !voiceEnabled, config.onVoiceBackendSelect));
        voiceModelCombo.update(selection("settings:intelligence-voice-model", voice.models, voice.model,
                                         !voiceEnabled, config.onVoiceModelSelect));
        voiceNameCombo.update(selection("settings:intelligence-voice-name", voice.voices, voice.voice,
                                        !voiceEnabled, config.onVoiceSelect));
        voiceStatus.update(voice.status);
        voiceGroup.update({
            .title = "Voice",
            .description = "Talk to the agent through a live voice model.",
            .rows = {
                {
                    .label = "Status",
                    .description = "The backend provider supplies the endpoint and API key.",
                    .control = &voiceStatus,
                },
                {
                    .label = "Backend Model",
                    .description = "Handles reasoning and tool calls.",
                    .control = &voiceBackendCombo,
                },
                {
                    .label = "Voice Model",
                    .description = "Model that listens and speaks.",
                    .control = &voiceModelCombo,
                },
                {
                    .label = "Voice",
                    .description = "Voice used for spoken replies.",
                    .control = &voiceNameCombo,
                },
            },
        });
    }

    static Sakura::Combo::Config selection(std::string id,
                                           const std::vector<std::string>& options,
                                           std::optional<U64> selected,
                                           bool disabled,
                                           const std::function<void(U64)>& onSelect) {
        return {
            .id = std::move(id),
            .options = options,
            .value = selected.has_value() && *selected < options.size() ? options[*selected] : "",
            .disabled = disabled,
            .selectedIndex = selected,
            .onSelect = [onSelect](U64 index) {
                if (onSelect) {
                    onSelect(index);
                }
            },
            .borderColorKey = "border",
        };
    }

    Config config;
    SettingsHeader header;
    Sakura::Toggle enabledToggle;
    SectionGroup assistantGroup;
    std::vector<std::unique_ptr<ProviderControls>> providerControls;
    CompactButton addProviderButton;
    SectionGroup providersGroup;
    std::vector<std::unique_ptr<ModelControls>> modelControls;
    CompactButton addModelButton;
    SectionGroup modelsGroup;
    Sakura::Combo voiceBackendCombo;
    Sakura::Combo voiceModelCombo;
    Sakura::Combo voiceNameCombo;
    StatusBadge voiceStatus;
    SectionGroup voiceGroup;
    IntelligenceProviderView providerPage;
    IntelligenceModelView modelPage;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_INTELLIGENCE_HH
