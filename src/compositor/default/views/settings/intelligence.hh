#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_INTELLIGENCE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_INTELLIGENCE_HH

#include "components/settings_group.hh"
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
        std::vector<std::string> tags;
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
        IntelligencePage page = IntelligencePage::Overview;
        IntelligenceProviderPage::Config provider;
        IntelligenceModelPage::Config model;
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

        if (this->config.enabled && this->config.page == IntelligencePage::Provider) {
            providerPage.update(this->config.provider);
            stack.update({.items = {{.child = &providerPage.content()}}});
            return;
        }
        if (this->config.enabled && this->config.page == IntelligencePage::AddModel) {
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
        Sakura::Badge status;
        Sakura::Button edit;
        Sakura::Button remove;
        Sakura::Stack row;
    };

    struct ModelControls {
        std::vector<std::unique_ptr<Sakura::Badge>> tags;
        Sakura::Button remove;
        Sakura::Stack row;
    };

    void updateProviders() {
        while (providerControls.size() < config.providers.size()) {
            providerControls.push_back(std::make_unique<ProviderControls>());
        }
        std::vector<Sakura::SettingCard::Row> rows;
        rows.reserve(config.providers.size());
        for (U64 index = 0; index < config.providers.size(); ++index) {
            const auto& provider = config.providers[index];
            auto& controls = *providerControls[index];
            controls.status.update(statusBadge(provider.status, provider.ready, provider.pending));
            controls.edit.update(compactButton("Edit", [this, name = provider.name]() {
                if (config.onEditProvider) {
                    config.onEditProvider(name);
                }
            }));
            controls.remove.update(compactButton("Remove", [this, name = provider.name]() {
                if (config.onRemoveProvider) {
                    config.onRemoveProvider(name);
                }
            }));
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
        addProviderButton.update(primaryButton([this]() {
            if (config.onOpenAddProvider) {
                config.onOpenAddProvider();
            }
        }));
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
            controls.remove.update(compactButton("Remove", [this, model]() {
                if (config.onRemoveModel) {
                    config.onRemoveModel(model);
                }
            }));
            while (controls.tags.size() < model.tags.size()) {
                controls.tags.push_back(std::make_unique<Sakura::Badge>());
            }
            std::vector<Sakura::Stack::Item> items;
            for (U64 tag = 0; tag < model.tags.size(); ++tag) {
                controls.tags[tag]->update({
                    .label = model.tags[tag],
                    .backgroundColorKey = "header_hovered",
                    .borderColorKey = "border",
                    .textColorKey = "text_secondary",
                });
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
        addModelButton.update(primaryButton([this]() {
            if (config.onOpenAddModel) {
                config.onOpenAddModel();
            }
        }));
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

        voiceBackendCombo.update({
            .id = "settings:intelligence-voice-backend",
            .options = voice.backends,
            .value = voice.backend.has_value() && *voice.backend < voice.backends.size()
                         ? voice.backends[*voice.backend]
                         : "",
            .disabled = !voiceEnabled,
            .selectedIndex = voice.backend,
            .onSelect = [this](U64 index) {
                if (config.onVoiceBackendSelect) {
                    config.onVoiceBackendSelect(index);
                }
            },
            .borderColorKey = "border",
        });
        voiceModelCombo.update({
            .id = "settings:intelligence-voice-model",
            .options = voice.models,
            .value = voice.model,
            .disabled = !voiceEnabled,
            .onChange = [this](const std::string& value) {
                if (config.onVoiceModelChange) {
                    config.onVoiceModelChange(value);
                }
            },
            .borderColorKey = "border",
        });
        voiceNameCombo.update({
            .id = "settings:intelligence-voice-name",
            .options = voice.voices,
            .value = voice.voice,
            .disabled = !voiceEnabled,
            .onChange = [this](const std::string& value) {
                if (config.onVoiceChange) {
                    config.onVoiceChange(value);
                }
            },
            .borderColorKey = "border",
        });
        voiceStatus.update(statusBadge(voice.status, voice.available && voice.ready,
                                       !voice.available || voice.pending));
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

    static Sakura::Badge::Config statusBadge(const std::string& label, bool ready, bool neutral) {
        if (!ready && neutral) {
            return {
                .label = label,
                .backgroundColorKey = "header_hovered",
                .borderColorKey = "border",
                .textColorKey = "text_secondary",
            };
        }
        const std::string tone = ready ? "banner_success" : "banner_warning";
        return {
            .label = label,
            .backgroundColorKey = tone + "_bg",
            .borderColorKey = tone + "_border",
            .textColorKey = tone + "_text",
        };
    }

    static Sakura::Button::Config compactButton(std::string label, std::function<void()> onClick) {
        return {
            .str = std::move(label),
            .textScale = 0.9f,
            .padding = {9.0f, 5.0f},
            .onClick = std::move(onClick),
            .font = Sakura::Text::Font::Sans,
        };
    }

    static Sakura::Button::Config primaryButton(std::function<void()> onClick) {
        return {
            .str = "Add",
            .colorKey = "primary_btn",
            .hoveredColorKey = "primary_btn_hovered",
            .activeColorKey = "primary_btn_hovered",
            .borderColorKey = "primary_btn",
            .textColorKey = "primary_btn_text",
            .padding = {14.0f, 5.0f},
            .onClick = std::move(onClick),
            .font = Sakura::Text::Font::Sans,
        };
    }

    Config config;
    SettingsHeader header;
    Sakura::Toggle enabledToggle;
    SettingsGroup assistantGroup;
    std::vector<std::unique_ptr<ProviderControls>> providerControls;
    Sakura::Button addProviderButton;
    SettingsGroup providersGroup;
    std::vector<std::unique_ptr<ModelControls>> modelControls;
    Sakura::Button addModelButton;
    SettingsGroup modelsGroup;
    Sakura::Combo voiceBackendCombo;
    Sakura::Combo voiceModelCombo;
    Sakura::Combo voiceNameCombo;
    Sakura::Badge voiceStatus;
    SettingsGroup voiceGroup;
    IntelligenceProviderPage providerPage;
    IntelligenceModelPage modelPage;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_INTELLIGENCE_HH
