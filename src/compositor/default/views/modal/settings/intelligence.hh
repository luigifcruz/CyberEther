#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_INTELLIGENCE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_INTELLIGENCE_HH

#include "components/action_table.hh"

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
        std::string status;
        bool ready = false;
        bool pending = false;
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
        using Text = Sakura::Text;

        this->config = std::move(config);

        title.update({
            .str = "Intelligence",
            .font = Text::Font::Bold,
            .scale = 1.2f,
        });

        description.update({
            .str = "Configure intelligence features and model access.",
            .tone = Text::Tone::Secondary,
            .wrapped = true,
        });

        divider.update({});

        enabledField.update({
            .description = "Hide all AI features and disconnect from model providers.",
            .control = &enabledCheckbox,
            .divider = this->config.enabled,
        });

        enabledCheckbox.update({
            .label = "Enable intelligence features",
            .value = this->config.enabled,
            .onChange = [this](bool value) {
                if (this->config.onEnabledChange) {
                    this->config.onEnabledChange(value);
                }
            },
        });

        providersTitle.update({.str = "Providers"});

        providersDescription.update({
            .str = "OpenAI-compatible endpoints the agent can use.",
            .tone = Text::Tone::Secondary,
            .wrapped = true,
        });

        std::vector<ActionTable::Row> providerRows;
        providerRows.reserve(this->config.providers.size());
        for (const auto& provider : this->config.providers) {
            providerRows.push_back({
                .cells = {
                    {.str = provider.name},
                    {
                        .str = provider.status,
                        .tone = provider.ready     ? Text::Tone::Success
                              : provider.pending ? Text::Tone::Disabled
                                                 : Text::Tone::Warning,
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
            .str = "Add Provider",
            .size = {Sakura::Fill, 34.0f},
            .onClick = [this]() {
                if (this->config.onOpenAddProvider) {
                    this->config.onOpenAddProvider();
                }
            },
        });

        modelsDivider.update({});

        modelsTitle.update({.str = "Models"});

        modelsDescription.update({
            .str = "Models available to the agent chat.",
            .tone = Text::Tone::Secondary,
            .wrapped = true,
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
                        .tone = model.context.empty() ? Text::Tone::Disabled : Text::Tone::Primary,
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
            .str = "Add Model",
            .size = {Sakura::Fill, 34.0f},
            .onClick = [this]() {
                if (this->config.onOpenAddModel) {
                    this->config.onOpenAddModel();
                }
            },
        });

        voiceDivider.update({});

        voiceTitle.update({.str = "Voice"});

        voiceDescription.update({
            .str = "The voice model listens and speaks with the chosen voice, and hands reasoning and tool calls "
                   "to the backend model. Only saved models from providers with live voice support can be the "
                   "backend, and that provider supplies the endpoint and API key.",
            .tone = Text::Tone::Secondary,
            .wrapped = true,
        });

        const auto& voice = this->config.voice;
        const bool voiceEnabled = voice.available && !voice.backends.empty();

        voiceStatusText.update({
            .str = voice.status,
            .tone = !voice.available || voice.pending ? Text::Tone::Disabled
                  : voice.ready                       ? Text::Tone::Success
                                                      : Text::Tone::Warning,
            .wrapped = true,
        });

        voiceBackendCombo.update({
            .id = "settings:intelligence-voice-backend",
            .options = voice.backends,
            .value = voice.backend.has_value() && *voice.backend < voice.backends.size()
                         ? voice.backends[*voice.backend]
                         : "",
            .disabled = !voiceEnabled,
            .selectedIndex = voice.backend,
            .onSelect = [this](U64 index) {
                if (this->config.onVoiceBackendSelect) {
                    this->config.onVoiceBackendSelect(index);
                }
            },
        });

        voiceModelCombo.update({
            .id = "settings:intelligence-voice-model",
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
            .id = "settings:intelligence-voice-name",
            .options = voice.voices,
            .value = voice.voice,
            .disabled = !voiceEnabled,
            .onChange = [this](const std::string& value) {
                if (this->config.onVoiceChange) {
                    this->config.onVoiceChange(value);
                }
            },
        });

        voiceTable.update({
            .columns = {
                "Backend Model",
                "Voice Model",
                "Voice",
                "Status",
            },
            .cellRows = {
                {&voiceBackendCombo, &voiceModelCombo, &voiceNameCombo, &voiceStatusText},
            },
            .fixedColumnWidths = {
                0.0f,
                150.0f,
                130.0f,
                120.0f,
            },
            .wrapped = true,
        });

        std::vector<Sakura::Stack::Item> items = {
            {.child = &title},
            {.child = &description},
            {.child = &divider},
            {.child = &enabledField},
        };
        if (this->config.enabled) {
            items.push_back({.child = &providersTitle});
            items.push_back({.child = &providersTable});
            items.push_back({.child = &addProviderButton});
            items.push_back({.child = &providersDescription});
            items.push_back({.child = &modelsDivider});
            items.push_back({.child = &modelsTitle});
            items.push_back({.child = &modelsTable});
            items.push_back({.child = &addModelButton});
            items.push_back({.child = &modelsDescription});
            items.push_back({.child = &voiceDivider});
            items.push_back({.child = &voiceTitle});
            items.push_back({.child = &voiceTable});
            items.push_back({.child = &voiceDescription});
        }
        stack.update({.items = std::move(items)});
    }

    Sakura::Component& content() {
        return stack;
    }

 private:
    Config config;
    Sakura::Text title;
    Sakura::Text description;
    Sakura::Divider divider;
    Sakura::Checkbox enabledCheckbox;
    Sakura::SettingField enabledField;
    Sakura::Text providersTitle;
    Sakura::Text providersDescription;
    ActionTable providersTable;
    Sakura::Button addProviderButton;
    Sakura::Divider modelsDivider;
    Sakura::Text modelsTitle;
    Sakura::Text modelsDescription;
    ActionTable modelsTable;
    Sakura::Button addModelButton;
    Sakura::Divider voiceDivider;
    Sakura::Text voiceTitle;
    Sakura::Text voiceDescription;
    Sakura::Text voiceStatusText;
    Sakura::Combo voiceBackendCombo;
    Sakura::Combo voiceModelCombo;
    Sakura::Combo voiceNameCombo;
    Sakura::Table voiceTable;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_INTELLIGENCE_HH
