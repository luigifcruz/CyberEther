#include "intelligence.hh"

namespace Jetstream {

void IntelligenceSettingsView::update(Config config) {
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
        updateMcp();
        items.push_back({.child = &providersGroup});
        items.push_back({.child = &modelsGroup});
        items.push_back({.child = &voiceGroup});
        items.push_back({.child = &mcpGroup});
    }
    stack.update({
        .spacing = Sakura::Spacing::XSmall,
        .items = std::move(items),
    });
}

Sakura::Component& IntelligenceSettingsView::content() {
    return stack;
}

Sakura::Callback<void()> IntelligenceSettingsView::back() {
    return [this]() {
        if (config.onNavigate) {
            config.onNavigate(IntelligencePage::Overview);
        }
    };
}

void IntelligenceSettingsView::updateProviders() {
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
            .icon = ICON_LC_PENCIL,
            .onClick = [this, name = provider.name]() {
                if (config.onEditProvider) {
                    config.onEditProvider(name);
                }
            },
        });
        controls.remove.update({
            .str = "Remove",
            .icon = ICON_LC_TRASH_2,
            .tone = CompactButton::Tone::Danger,
            .onClick = [this, name = provider.name]() {
                if (config.onRemoveProvider) {
                    config.onRemoveProvider(name);
                }
            },
        });
        controls.row.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = Sakura::Spacing::Small,
            .items = {
                {.child = &controls.status},
                {.child = &controls.edit},
                {.child = &controls.remove},
            },
        });
        rows.push_back({.label = provider.name, .control = &controls.row, .wide = true});
    }
    addProviderButton.update({
        .str = "Add",
        .icon = ICON_LC_PLUS,
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

void IntelligenceSettingsView::updateModels() {
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
            .icon = ICON_LC_TRASH_2,
            .tone = CompactButton::Tone::Danger,
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
            .spacing = Sakura::Spacing::Small,
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
        .icon = ICON_LC_PLUS,
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

void IntelligenceSettingsView::updateVoice() {
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

void IntelligenceSettingsView::updateMcp() {
    const auto& mcp = config.mcp;
    mcpToggle.update({
        .value = mcp.settings.enabled,
        .onChange = [this](bool value) {
            auto settings = config.mcp.settings;
            settings.enabled = value;
            changeMcp(settings);
        },
    });
    mcpStatus.update({
        .label = !mcp.url.empty() ? "Listening" : !mcp.error.empty() ? "Failed" : "Stopped",
        .tone = !mcp.url.empty()   ? StatusBadge::Tone::Success
              : !mcp.error.empty() ? StatusBadge::Tone::Warning
                                   : StatusBadge::Tone::Neutral,
    });
    mcpAddressInput.update({
        .icon = ICON_LC_NETWORK,
        .value = mcp.settings.address,
        .onChange = [this](const std::string& value) {
            auto settings = config.mcp.settings;
            settings.address = value.empty() ? Settings::Intelligence::Mcp{}.address : value;
            changeMcp(settings);
        },
    });
    mcpPortInput.update({
        .icon = ICON_LC_ETHERNET_PORT,
        .value = jst::fmt::format("{}", mcp.settings.port),
        .onChange = [this](const std::string& value) {
            U16 port = 0;
            if (portValue(value, port)) {
                auto settings = config.mcp.settings;
                settings.port = port;
                changeMcp(settings);
            }
        },
    });
    mcpGroup.update({
        .title = "MCP Server",
        .description = "Lets external agents connect over the Model Context Protocol and use the same tools as "
                       "the built-in assistant. Each call acts on the flowgraph focused in this window.",
        .warning = "Do not expose the MCP server directly to the internet.",
        .rows = {
            {
                .label = "Enable MCP Server",
                .description = "Serve the agent tools at the address and port below.",
                .control = &mcpToggle,
            },
            {
                .label = "Status",
                .description = !mcp.url.empty()   ? "Listening on " + mcp.url
                             : !mcp.error.empty() ? mcp.error
                                                  : "The server is not running.",
                .control = &mcpStatus,
            },
            {
                .label = "Address",
                .description = "The network address the server listens on. Use 0.0.0.0 to accept clients on "
                               "every interface.",
                .control = &mcpAddressInput,
            },
            {
                .label = "Port",
                .description = "The port that serves MCP clients.",
                .control = &mcpPortInput,
            },
        },
    });
}

void IntelligenceSettingsView::changeMcp(const Settings::Intelligence::Mcp& settings) const {
    if (config.onMcpChange) {
        config.onMcpChange(settings);
    }
}

bool IntelligenceSettingsView::portValue(const std::string& label, U16& port) {
    if (label.empty() || label.size() > 5 || !std::all_of(label.begin(), label.end(), [](unsigned char c) {
        return std::isdigit(c);
    })) {
        return false;
    }
    const U64 value = std::stoull(label);
    if (value == 0 || value > 65535) {
        return false;
    }
    port = static_cast<U16>(value);
    return true;
}

Sakura::Combo::Config IntelligenceSettingsView::selection(std::string id,
                                                          const std::vector<std::string>& options,
                                                          std::optional<U64> selected,
                                                          bool disabled,
                                                          const Sakura::Callback<void(U64)>& onSelect) {
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
    };
}

}  // namespace Jetstream
