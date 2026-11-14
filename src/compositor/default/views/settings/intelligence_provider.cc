#include "intelligence_provider.hh"

namespace Jetstream {

void IntelligenceProviderView::update(Config config) {
    this->config = std::move(config);

    if (this->config.session != lastSession) {
        lastSession = this->config.session;
        apiKey.clear();
        applyKeySource(this->config.editMode ? this->config.editProvider : Settings::Intelligence::Provider{});
        if (this->config.editMode) {
            applyProvider(this->config.editProvider);
        } else {
            reset();
        }
    }

    const std::string title = this->config.editMode ? "Edit Provider" : "Add Provider";
    breadcrumb.update({
        .root = "Intelligence",
        .current = this->config.editMode ? this->config.editProvider.name : title,
        .onRoot = this->config.onBack,
    });
    header.update({
        .title = title,
        .description = "Connect an OpenAI-compatible endpoint for agent chat.",
    });

    std::vector<std::string> presetLabels;
    presetLabels.reserve(this->config.presets.size());
    for (const auto& preset : this->config.presets) {
        presetLabels.push_back(preset.label);
    }
    typeCombo.update({
        .id = "intelligence-provider:type",
        .options = std::move(presetLabels),
        .value = typeLabel,
        .disabled = this->config.saving,
        .onChange = [this](const std::string& value) {
            if (typeLabel == value) {
                return;
            }
            typeLabel = value;
            applyPreset(value);
            markDirty();
        },
    });
    nameInput.update({
        .icon = ICON_LC_TAG,
        .value = name,
        .hint = "Local Ollama",
        .submit = Sakura::TextInput::Submit::OnEdit,
        .disabled = this->config.saving,
        .onChange = [this](const std::string& value) {
            name = value;
            markDirty();
        },
    });
    hostInput.update({
        .icon = ICON_LC_GLOBE,
        .value = host,
        .hint = "http://localhost:11434",
        .submit = Sakura::TextInput::Submit::OnEdit,
        .disabled = this->config.saving,
        .onChange = [this](const std::string& value) {
            host = value;
            markDirty();
        },
    });
    endpointGroup.update({
        .title = "Endpoint",
        .description = "Where the provider lives and what to call it.",
        .rows = {
            {
                .label = "Provider Type",
                .description = typeDescription(),
                .control = &typeCombo,
            },
            {
                .label = "Name",
                .description = "Display name for this provider.",
                .control = &nameInput,
            },
            {
                .label = "Base URL",
                .description = "Root address of the provider API.",
                .control = &hostInput,
            },
        },
    });

    apiTabs.update({
        .items = {{.label = ChatCompletionsLabel}, {.label = ResponsesLabel}},
        .selected = api == "responses" ? 1u : 0u,
        .compact = true,
        .track = true,
        .onSelect = [this](U64 index) {
            const std::string next = index == 1 ? "responses" : "chat_completions";
            if (this->config.saving || api == next) {
                return;
            }
            api = next;
            markDirty();
        },
    });
    chatPathInput.update({
        .icon = ICON_LC_ROUTE,
        .value = chatPath,
        .hint = this->config.chatPathHint,
        .submit = Sakura::TextInput::Submit::OnEdit,
        .disabled = this->config.saving,
        .onChange = [this](const std::string& value) {
            chatPath = value;
            markDirty();
        },
    });
    modelsPathInput.update({
        .icon = ICON_LC_ROUTE,
        .value = modelsPath,
        .hint = this->config.modelsPathHint,
        .submit = Sakura::TextInput::Submit::OnEdit,
        .disabled = this->config.saving,
        .onChange = [this](const std::string& value) {
            modelsPath = value;
            markDirty();
        },
    });
    keySourceTabs.update({
        .items = {{.icon = ICON_LC_KEY_ROUND, .label = KeychainLabel}, {.icon = ICON_LC_VARIABLE, .label = EnvironmentLabel}},
        .selected = keySource == KeySource::Environment ? 1u : 0u,
        .compact = true,
        .track = true,
        .onSelect = [this](U64 index) {
            const auto next = index == 1 ? KeySource::Environment : KeySource::Keychain;
            if (this->config.saving || keySource == next) {
                return;
            }
            keySource = next;
            markDirty();
        },
    });
    keySourceInput.update({
        .icon = ICON_LC_VARIABLE,
        .value = apiKeyEnv,
        .hint = "OPENROUTER_API_KEY",
        .submit = Sakura::TextInput::Submit::OnEdit,
        .disabled = this->config.saving,
        .onChange = [this](const std::string& value) {
            apiKeyEnv = value;
            markDirty();
        },
    });
    keyInput.update({
        .icon = ICON_LC_KEY_ROUND,
        .value = apiKey,
        .hint = keyHint(),
        .submit = Sakura::TextInput::Submit::OnEdit,
        .password = true,
        .disabled = this->config.saving,
        .onChange = [this](const std::string& value) {
            apiKey = value;
            markDirty();
        },
    });
    apiGroup.update({
        .title = "API",
        .description = "Request format, endpoints, and credentials.",
        .rows = {
            {
                .label = "Request Format",
                .description = "The API the endpoint speaks.",
                .control = &apiTabs,
                .wide = true,
            },
            {
                .label = "Chat Path",
                .description = "Chat endpoint relative to the base URL.",
                .control = &chatPathInput,
            },
            {
                .label = "Models Path",
                .description = "Model list endpoint relative to the base URL.",
                .control = &modelsPathInput,
            },
            {
                .label = "Key Source",
                .description = "Where the API key comes from.",
                .control = &keySourceTabs,
                .wide = true,
            },
            {
                .label = keySource == KeySource::Environment ? "Variable Name" : "API Key",
                .description = keyDescription(),
                .control = keySource == KeySource::Environment ? static_cast<Sakura::Component*>(&keySourceInput)
                                                               : static_cast<Sakura::Component*>(&keyInput),
            },
        },
    });

    testStatus.update(statusBadge());
    testButton.update({
        .str = this->config.testRunning ? "Testing..." : "Test",
        .icon = ICON_LC_PLUG,
        .disabled = !testEnabled(),
        .onClick = [this]() {
            if (testEnabled() && this->config.onTest) {
                this->config.onTest(buildProvider(), keychainKey(), draftRevision);
            }
        },
    });
    std::vector<Sakura::Stack::Item> testItems;
    if (!statusBadgeLabel().empty()) {
        testItems.push_back({.child = &testStatus});
    }
    testItems.push_back({.child = &testButton});
    testControls.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = Sakura::Spacing::Small,
        .items = std::move(testItems),
    });
    saveButton.update({
        .str = saveLabel(),
        .size = {Sakura::Fill, 40.0f},
        .variant = Sakura::Button::Variant::Action,
        .disabled = !saveEnabled(),
        .rounding = Sakura::Rounding::Card,
        .onClick = [this]() {
            if (saveEnabled() && this->config.onAdd) {
                this->config.onAdd(buildProvider(), keychainKey());
            }
        },
        .font = Sakura::Text::Font::SansBold,
    });
    connectionGroup.update({
        .title = "Connection",
        .description = "A successful test is required before saving.",
        .rows = {{
            .label = "Connection Test",
            .description = testDescription(),
            .control = &testControls,
            .wide = true,
        }},
        .footer = &saveButton,
    });

    stack.update({
        .spacing = Sakura::Spacing::XSmall,
        .items = {
            {.child = &breadcrumb},
            {.child = &header},
            {.child = &endpointGroup},
            {.child = &apiGroup},
            {.child = &connectionGroup},
        },
    });
}

Sakura::Component& IntelligenceProviderView::content() {
    return stack;
}

void IntelligenceProviderView::applyPreset(const std::string& label) {
    for (const auto& preset : config.presets) {
        if (preset.label == label) {
            applyProvider(preset.provider);
            applyKeySource(preset.provider);
            typeLabel = label;
            return;
        }
    }
}

void IntelligenceProviderView::applyProvider(const Settings::Intelligence::Provider& provider) {
    typeLabel = config.editPreset;
    name = provider.name;
    host = provider.host;
    api = provider.api;
    chatPath = provider.chatPath;
    modelsPath = provider.modelsPath;
    apiKey.clear();
    requestReasoning = provider.requestReasoning;
    markDirty();
}

void IntelligenceProviderView::markDirty() {
    ++draftRevision;
}

void IntelligenceProviderView::reset() {
    applyPreset(config.defaultPreset);
}

Settings::Intelligence::Provider IntelligenceProviderView::buildProvider() const {
    Settings::Intelligence::Provider provider;
    provider.name = name;
    provider.host = host;
    provider.api = api;
    if (!chatPath.empty()) {
        provider.chatPath = chatPath;
    }
    if (!modelsPath.empty()) {
        provider.modelsPath = modelsPath;
    }
    if (keySource == KeySource::Environment) {
        provider.apiKeyEnv = apiKeyEnv;
    }
    provider.requestReasoning = requestReasoning;
    return provider;
}

void IntelligenceProviderView::applyKeySource(const Settings::Intelligence::Provider& provider) {
    apiKeyEnv = provider.apiKeyEnv;
    keySource = !apiKeyEnv.empty() ? KeySource::Environment : KeySource::Keychain;
}

std::string IntelligenceProviderView::keychainKey() const {
    return keySource == KeySource::Keychain ? apiKey : "";
}

bool IntelligenceProviderView::keySourceValid() const {
    return keySource != KeySource::Environment || !apiKeyEnv.empty();
}

std::string IntelligenceProviderView::keyDescription() const {
    if (keySource == KeySource::Environment) {
        return "Externally managed. The key is read from this environment variable on each request "
               "and never saved.";
    }
    return "Saved in the system keychain, never in the settings file. Leave blank "
           "for a custom endpoint without authentication.";
}

std::string IntelligenceProviderView::keyHint() const {
    if (config.editMode && !config.editProvider.apiKeyId.empty()) {
        return "Stored in the keychain. Type to replace.";
    }
    return "API key";
}

bool IntelligenceProviderView::nameTaken() const {
    return std::find(config.existingNames.begin(), config.existingNames.end(), name) !=
           config.existingNames.end();
}

bool IntelligenceProviderView::testCurrent() const {
    return config.testCompleted && config.testRevision == draftRevision;
}

bool IntelligenceProviderView::testPassed() const {
    return testCurrent() && config.testSuccess;
}

bool IntelligenceProviderView::providerValid() const {
    return config.isValid && config.isValid(buildProvider()) && keySourceValid();
}

bool IntelligenceProviderView::testEnabled() const {
    return !config.saving && !config.testRunning && providerValid();
}

bool IntelligenceProviderView::saveEnabled() const {
    return !config.saving && !config.testRunning && testPassed() && providerValid();
}

std::string IntelligenceProviderView::saveLabel() const {
    if (config.saving) {
        return "Saving...";
    }
    if (config.editMode) {
        return "Save Provider";
    }
    return nameTaken() ? "Update Provider" : "Add Provider";
}

std::string IntelligenceProviderView::typeDescription() const {
    for (const auto& preset : config.presets) {
        if (preset.label == typeLabel) {
            return preset.description;
        }
    }
    return "";
}

std::string IntelligenceProviderView::testDescription() const {
    if (config.testRunning) {
        return "Testing...";
    }
    if (testCurrent() && !config.testMessage.empty()) {
        return config.testMessage;
    }
    return "Checks the endpoint and key.";
}

std::string IntelligenceProviderView::statusBadgeLabel() const {
    if (config.testRunning) {
        return "Testing";
    }
    if (testCurrent()) {
        return config.testSuccess ? "Passed" : "Failed";
    }
    return "";
}

StatusBadge::Config IntelligenceProviderView::statusBadge() const {
    if (!testCurrent()) {
        return {.label = statusBadgeLabel()};
    }
    return {
        .label = statusBadgeLabel(),
        .tone = config.testSuccess ? StatusBadge::Tone::Success : StatusBadge::Tone::Warning,
    };
}

}  // namespace Jetstream
