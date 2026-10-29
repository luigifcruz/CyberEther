#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_INTELLIGENCE_PROVIDER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_INTELLIGENCE_PROVIDER_HH

#include "../components/compact_button.hh"
#include "components/settings_breadcrumb.hh"
#include "../components/section_group.hh"
#include "components/settings_header.hh"
#include "../components/status_badge.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/settings.hh"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct IntelligenceProviderView {
    static constexpr const char* ChatCompletionsLabel = "Chat Completions";
    static constexpr const char* ResponsesLabel = "Responses";
    static constexpr const char* KeychainLabel = "System Keychain";
    static constexpr const char* EnvironmentLabel = "Environment Variable";

    struct Preset {
        std::string label;
        std::string description;
        Settings::Intelligence::Provider provider;
    };

    struct Config {
        std::string defaultPreset;
        std::string chatPathHint;
        std::string modelsPathHint;
        U64 session = 0;
        bool editMode = false;
        Settings::Intelligence::Provider editProvider;
        std::string editPreset;
        std::vector<Preset> presets;
        bool saving = false;
        bool testRunning = false;
        bool testCompleted = false;
        bool testSuccess = false;
        U64 testRevision = 0;
        std::string testMessage;
        std::vector<std::string> existingNames;
        Sakura::Callback<bool(const Settings::Intelligence::Provider&)> isValid;
        Sakura::Callback<void(const Settings::Intelligence::Provider&, const std::string&, U64)> onTest;
        Sakura::Callback<void(const Settings::Intelligence::Provider&, const std::string&)> onAdd;
        Sakura::Callback<void()> onBack;
    };

    void update(Config config) {
        using TextInput = Sakura::TextInput;

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
            .borderColorKey = "border",
        });
        nameInput.update({
            .value = name,
            .hint = "Local Ollama",
            .submit = TextInput::Submit::OnEdit,
            .disabled = this->config.saving,
            .onChange = [this](const std::string& value) {
                name = value;
                markDirty();
            },
            .borderColorKey = "border",
        });
        hostInput.update({
            .value = host,
            .hint = "http://localhost:11434",
            .submit = TextInput::Submit::OnEdit,
            .disabled = this->config.saving,
            .onChange = [this](const std::string& value) {
                host = value;
                markDirty();
            },
            .borderColorKey = "border",
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

        apiCombo.update({
            .id = "intelligence-provider:api",
            .options = {ChatCompletionsLabel, ResponsesLabel},
            .value = api == "responses" ? ResponsesLabel : ChatCompletionsLabel,
            .disabled = this->config.saving,
            .onChange = [this](const std::string& value) {
                const std::string next = value == ResponsesLabel ? "responses" : "chat_completions";
                if (api == next) {
                    return;
                }
                api = next;
                markDirty();
            },
            .borderColorKey = "border",
        });
        chatPathInput.update({
            .value = chatPath,
            .hint = this->config.chatPathHint,
            .submit = TextInput::Submit::OnEdit,
            .disabled = this->config.saving,
            .onChange = [this](const std::string& value) {
                chatPath = value;
                markDirty();
            },
            .borderColorKey = "border",
        });
        modelsPathInput.update({
            .value = modelsPath,
            .hint = this->config.modelsPathHint,
            .submit = TextInput::Submit::OnEdit,
            .disabled = this->config.saving,
            .onChange = [this](const std::string& value) {
                modelsPath = value;
                markDirty();
            },
            .borderColorKey = "border",
        });
        keySourceCombo.update({
            .id = "intelligence-provider:key-source",
            .options = {KeychainLabel, EnvironmentLabel},
            .value = keySourceLabel(),
            .disabled = this->config.saving,
            .onChange = [this](const std::string& value) {
                const auto next = value == EnvironmentLabel ? KeySource::Environment : KeySource::Keychain;
                if (keySource == next) {
                    return;
                }
                keySource = next;
                markDirty();
            },
            .borderColorKey = "border",
        });
        keySourceInput.update({
            .value = apiKeyEnv,
            .hint = "OPENROUTER_API_KEY",
            .submit = TextInput::Submit::OnEdit,
            .disabled = this->config.saving,
            .onChange = [this](const std::string& value) {
                apiKeyEnv = value;
                markDirty();
            },
            .borderColorKey = "border",
        });
        keyInput.update({
            .value = apiKey,
            .hint = keyHint(),
            .submit = TextInput::Submit::OnEdit,
            .password = true,
            .disabled = this->config.saving,
            .onChange = [this](const std::string& value) {
                apiKey = value;
                markDirty();
            },
            .borderColorKey = "border",
        });
        apiGroup.update({
            .title = "API",
            .description = "Request format, endpoints, and credentials.",
            .rows = {
                {
                    .label = "Request Format",
                    .description = "The API the endpoint speaks.",
                    .control = &apiCombo,
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
                    .control = &keySourceCombo,
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

    Sakura::Component& content() {
        return stack;
    }

 private:
    void applyPreset(const std::string& label) {
        for (const auto& preset : config.presets) {
            if (preset.label == label) {
                applyProvider(preset.provider);
                applyKeySource(preset.provider);
                typeLabel = label;
                return;
            }
        }
    }

    void applyProvider(const Settings::Intelligence::Provider& provider) {
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

    void markDirty() {
        ++draftRevision;
    }

    void reset() {
        applyPreset(config.defaultPreset);
    }

    Settings::Intelligence::Provider buildProvider() const {
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

    void applyKeySource(const Settings::Intelligence::Provider& provider) {
        apiKeyEnv = provider.apiKeyEnv;
        keySource = !apiKeyEnv.empty() ? KeySource::Environment : KeySource::Keychain;
    }

    std::string keySourceLabel() const {
        return keySource == KeySource::Environment ? EnvironmentLabel : KeychainLabel;
    }

    std::string keychainKey() const {
        return keySource == KeySource::Keychain ? apiKey : "";
    }

    bool keySourceValid() const {
        return keySource != KeySource::Environment || !apiKeyEnv.empty();
    }

    std::string keyDescription() const {
        if (keySource == KeySource::Environment) {
            return "Externally managed. The key is read from this environment variable on each request "
                   "and never saved.";
        }
        return "Saved in the system keychain, never in the settings file. Leave blank "
               "for a custom endpoint without authentication.";
    }

    std::string keyHint() const {
        if (config.editMode && !config.editProvider.apiKeyId.empty()) {
            return "Stored in the keychain. Type to replace.";
        }
        return "API key";
    }

    bool nameTaken() const {
        return std::find(config.existingNames.begin(), config.existingNames.end(), name) !=
               config.existingNames.end();
    }

    bool testCurrent() const {
        return config.testCompleted && config.testRevision == draftRevision;
    }

    bool testPassed() const {
        return testCurrent() && config.testSuccess;
    }

    bool providerValid() const {
        return config.isValid && config.isValid(buildProvider()) && keySourceValid();
    }

    bool testEnabled() const {
        return !config.saving && !config.testRunning && providerValid();
    }

    bool saveEnabled() const {
        return !config.saving && !config.testRunning && testPassed() && providerValid();
    }

    std::string saveLabel() const {
        if (config.saving) {
            return "Saving...";
        }
        if (config.editMode) {
            return "Save Provider";
        }
        return nameTaken() ? "Update Provider" : "Add Provider";
    }

    std::string typeDescription() const {
        for (const auto& preset : config.presets) {
            if (preset.label == typeLabel) {
                return preset.description;
            }
        }
        return "";
    }

    std::string testDescription() const {
        if (config.testRunning) {
            return "Testing...";
        }
        if (testCurrent() && !config.testMessage.empty()) {
            return config.testMessage;
        }
        return "Checks the endpoint and key.";
    }

    std::string statusBadgeLabel() const {
        if (config.testRunning) {
            return "Testing";
        }
        if (testCurrent()) {
            return config.testSuccess ? "Passed" : "Failed";
        }
        return "";
    }

    StatusBadge::Config statusBadge() const {
        if (!testCurrent()) {
            return {.label = statusBadgeLabel()};
        }
        return {
            .label = statusBadgeLabel(),
            .tone = config.testSuccess ? StatusBadge::Tone::Success : StatusBadge::Tone::Warning,
        };
    }

    enum class KeySource {
        Keychain,
        Environment,
    };

    Config config;
    std::optional<U64> lastSession;
    U64 draftRevision = 0;
    bool requestReasoning = false;
    std::string typeLabel;
    std::string name;
    std::string host;
    std::string api;
    std::string chatPath;
    std::string modelsPath;
    std::string apiKey;
    std::string apiKeyEnv;
    KeySource keySource = KeySource::Keychain;
    SettingsBreadcrumb breadcrumb;
    SettingsHeader header;
    Sakura::Combo typeCombo;
    Sakura::TextInput nameInput;
    Sakura::TextInput hostInput;
    SectionGroup endpointGroup;
    Sakura::Combo apiCombo;
    Sakura::TextInput chatPathInput;
    Sakura::TextInput modelsPathInput;
    Sakura::Combo keySourceCombo;
    Sakura::TextInput keyInput;
    Sakura::TextInput keySourceInput;
    SectionGroup apiGroup;
    StatusBadge testStatus;
    CompactButton testButton;
    Sakura::Stack testControls;
    Sakura::Button saveButton;
    SectionGroup connectionGroup;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_INTELLIGENCE_PROVIDER_HH
