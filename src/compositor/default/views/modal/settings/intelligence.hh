#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_INTELLIGENCE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_INTELLIGENCE_HH

#include "components/action_table.hh"
#include "components/settings_group.hh"

#include "jetstream/logger.hh"
#include "jetstream/render/sakura/base.hh"

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct IntelligenceSettingsPanel {
    struct ProviderRow {
        std::string name;
        std::string status;
        bool ready = false;
        bool pending = false;
    };

    struct ModelRow {
        std::string provider;
        std::string id;
        std::string label;
        std::string context;
    };

    struct VoiceRow {
        bool available = false;
        std::vector<std::string> backends;
        std::optional<U64> backend;
        std::vector<std::string> models;
        std::string model;
        std::vector<std::string> voices;
        std::string voice;
    };

    struct Config {
        bool enabled = true;
        std::vector<ProviderRow> providers;
        std::vector<ModelRow> models;
        VoiceRow voice;
        std::function<void(bool)> onEnabledChange;
        std::function<void(U64)> onVoiceBackendSelect;
        std::function<void(const std::string&)> onVoiceModelChange;
        std::function<void(const std::string&)> onVoiceChange;
        std::function<void()> onOpenAddProvider;
        std::function<void(const std::string&)> onEditProvider;
        std::function<void(const std::string&)> onRemoveProvider;
        std::function<void()> onOpenAddModel;
        std::function<void(const ModelRow&)> onRemoveModel;
    };

    void update(Config config) {
        this->config = std::move(config);

        title.update({
            .id = "IntelligenceTitle",
            .str = "Intelligence",
            .font = Sakura::Text::Font::Bold,
            .scale = 1.2f,
        });

        description.update({
            .id = "IntelligenceDescription",
            .str = "Configure intelligence features and model access.",
            .tone = Sakura::Text::Tone::Secondary,
            .wrapped = true,
        });

        divider.update({
            .id = "IntelligenceHeaderDivider",
        });

        featuresSection.update({
            .id = "IntelligenceFeaturesSection",
            .title = "Features",
            .description = "Whether the assistant and other AI features are available.",
        });
        providersSection.update({
            .id = "IntelligenceProvidersSection",
            .title = "Providers",
            .description = "OpenAI-compatible endpoints the agent can use.",
        });
        modelsSection.update({
            .id = "IntelligenceModelsSection",
            .title = "Models",
            .description = "Models available to the agent chat.",
        });
        voiceSection.update({
            .id = "IntelligenceVoiceSection",
            .title = "Voice",
            .description = "The voice model listens and speaks with the chosen voice, and hands reasoning and tool "
                           "calls to the backend model. Only saved models from providers with live voice support "
                           "can be the backend, and that provider supplies the endpoint and API key.",
        });
        enabledField.update({
            .id = "IntelligenceEnabledField",
            .description = "Hide all AI features and disconnect from model providers.",
            .divider = false,
        });

        enabledCheckbox.update({
            .id = "IntelligenceEnabledCheckbox",
            .label = "Enable intelligence features",
            .value = this->config.enabled,
            .onChange = [this](bool value) {
                if (this->config.onEnabledChange) {
                    this->config.onEnabledChange(value);
                }
            },
        });

        std::vector<ActionTable::Row> providerRows;
        providerRows.reserve(this->config.providers.size());
        for (const auto& provider : this->config.providers) {
            providerRows.push_back({
                .cells = {
                    {.str = provider.name},
                    {
                        .str = provider.status,
                        .tone = provider.ready     ? Sakura::Text::Tone::Success
                              : provider.pending ? Sakura::Text::Tone::Disabled
                                                 : Sakura::Text::Tone::Warning,
                    },
                },
                .actions = {
                    {
                        .str = "Edit",
                        .onClick = [this, name = provider.name]() {
                            if (this->config.onEditProvider) {
                                this->config.onEditProvider(name);
                            }
                        },
                    },
                    {
                        .str = "Remove",
                        .onClick = [this, name = provider.name]() {
                            if (this->config.onRemoveProvider) {
                                this->config.onRemoveProvider(name);
                            }
                        },
                    },
                },
            });
        }
        providersTable.update({
            .id = "IntelligenceProvidersTable",
            .columns = {
                "Name",
                "Status",
                "Action",
            },
            .fixedColumnWidths = {
                150.0f,
                0.0f,
                130.0f,
            },
            .empty = "No providers configured. Add one to enable the agent.",
            .rows = std::move(providerRows),
        });

        addProviderButton.update({
            .id = "IntelligenceAddProvider",
            .str = "Add Provider",
            .size = {-1.0f, 34.0f},
            .onClick = [this]() {
                if (this->config.onOpenAddProvider) {
                    this->config.onOpenAddProvider();
                }
            },
        });

        std::vector<ActionTable::Row> modelRows;
        modelRows.reserve(this->config.models.size());
        for (const auto& model : this->config.models) {
            modelRows.push_back({
                .cells = {
                    {.str = model.provider},
                    {.str = model.label},
                    {
                        .str = model.context.empty() ? "Unknown" : model.context,
                        .tone = model.context.empty() ? Sakura::Text::Tone::Disabled
                                                      : Sakura::Text::Tone::Primary,
                        .wrapped = false,
                    },
                },
                .actions = {
                    {
                        .str = "Remove",
                        .onClick = [this, model]() {
                            if (this->config.onRemoveModel) {
                                this->config.onRemoveModel(model);
                            }
                        },
                    },
                },
            });
        }
        modelsTable.update({
            .id = "IntelligenceModelsTable",
            .columns = {
                "Provider",
                "Model",
                "Context",
                "Action",
            },
            .fixedColumnWidths = {
                150.0f,
                0.0f,
                80.0f,
                80.0f,
            },
            .empty = "No models configured. Add one to enable the agent.",
            .rows = std::move(modelRows),
        });

        addModelButton.update({
            .id = "IntelligenceAddModel",
            .str = "Add Model",
            .size = {-1.0f, 34.0f},
            .onClick = [this]() {
                if (this->config.onOpenAddModel) {
                    this->config.onOpenAddModel();
                }
            },
        });

        const auto& voice = this->config.voice;
        const bool voiceEnabled = voice.available && !voice.backends.empty();

        voiceTable.update({
            .id = "IntelligenceVoiceTable",
            .columns = {
                "Backend Model",
                "Voice Model",
                "Voice",
            },
            .fixedColumnWidths = {
                0.0f,
                150.0f,
                130.0f,
            },
            .wrapped = true,
        });

        voiceBackendCombo.update({
            .id = "##intelligence-voice-backend",
            .options = voice.backends,
            .value = voice.backend.has_value() ? voice.backends[*voice.backend] : "",
            .disabled = !voiceEnabled,
            .selectedIndex = voice.backend,
            .onSelect = [this](U64 index) {
                if (this->config.onVoiceBackendSelect) {
                    this->config.onVoiceBackendSelect(index);
                }
            },
        });

        voiceModelCombo.update({
            .id = "##intelligence-voice-model",
            .options = voice.models,
            .value = voice.model,
            .disabled = !voiceEnabled,
            .onChange = [this](const std::string& value) {
                if (this->config.onVoiceModelChange) {
                    this->config.onVoiceModelChange(value);
                }
            },
        });

        voiceNameCombo.update({
            .id = "##intelligence-voice-name",
            .options = voice.voices,
            .value = voice.voice,
            .disabled = !voiceEnabled,
            .onChange = [this](const std::string& value) {
                if (this->config.onVoiceChange) {
                    this->config.onVoiceChange(value);
                }
            },
        });
    }

    void render(const Sakura::Context& ctx) const {
        title.render(ctx);
        description.render(ctx);
        divider.render(ctx);

        featuresSection.render(ctx, [&](const Sakura::Context& ctx) {
            enabledField.render(ctx, [&](const Sakura::Context& ctx) {
                enabledCheckbox.render(ctx);
            });
        });
        if (!config.enabled) {
            return;
        }

        providersSection.render(ctx, [&](const Sakura::Context& ctx) {
            providersTable.render(ctx);
            addProviderButton.render(ctx);
        });

        modelsSection.render(ctx, [&](const Sakura::Context& ctx) {
            modelsTable.render(ctx);
            addModelButton.render(ctx);
        });

        voiceSection.render(ctx, [&](const Sakura::Context& ctx) {
            Sakura::Table::Row voiceRow;
            voiceRow.push_back([this](const Sakura::Context& ctx) {
                voiceBackendCombo.render(ctx);
            });
            voiceRow.push_back([this](const Sakura::Context& ctx) {
                voiceModelCombo.render(ctx);
            });
            voiceRow.push_back([this](const Sakura::Context& ctx) {
                voiceNameCombo.render(ctx);
            });
            Sakura::Table::Rows voiceRows;
            voiceRows.push_back(std::move(voiceRow));
            voiceTable.render(ctx, std::move(voiceRows));
        });
    }

 private:
    Config config;
    Sakura::Text title;
    Sakura::Text description;
    Sakura::Divider divider;
    SettingsGroup featuresSection;
    SettingsGroup providersSection;
    SettingsGroup modelsSection;
    SettingsGroup voiceSection;
    Sakura::SettingField enabledField;
    Sakura::Checkbox enabledCheckbox;
    ActionTable providersTable;
    Sakura::Button addProviderButton;
    ActionTable modelsTable;
    Sakura::Button addModelButton;
    Sakura::Table voiceTable;
    Sakura::Combo voiceBackendCombo;
    Sakura::Combo voiceModelCombo;
    Sakura::Combo voiceNameCombo;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_INTELLIGENCE_HH
