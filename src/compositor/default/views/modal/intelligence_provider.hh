#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_INTELLIGENCE_PROVIDER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_INTELLIGENCE_PROVIDER_HH

#include "../components/modal_header.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_fa.hh"
#include "jetstream/settings.hh"

#include <algorithm>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct IntelligenceProviderView {
    static constexpr const char* ChatCompletionsLabel = "Chat Completions";
    static constexpr const char* ResponsesLabel = "Responses";

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
        std::function<bool(const Settings::Intelligence::Provider&)> isValid;
        std::function<void(const Settings::Intelligence::Provider&, const std::string&, U64)> onTest;
        std::function<void(const Settings::Intelligence::Provider&, const std::string&)> onAdd;
    };

    void update(Config config) {
        using Button = Sakura::Button;
        using TextInput = Sakura::TextInput;

        this->config = std::move(config);

        if (this->config.session != lastSession) {
            lastSession = this->config.session;
            apiKey.clear();
            if (this->config.editMode) {
                applyProvider(this->config.editProvider);
            } else {
                reset();
            }
        }

        header.update({
            .title = this->config.editMode ? ICON_FA_ROBOT " Edit Provider" : ICON_FA_ROBOT " Add Provider",
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

        typeField.update({
            .label = "Provider Type",
            .description = typeDescription(),
            .control = &typeCombo,
        });

        nameInput.update({
            .value = name,
            .hint = "Name (e.g. Local Ollama).",
            .submit = TextInput::Submit::OnEdit,
            .disabled = this->config.saving,
            .onChange = [this](const std::string& value) {
                name = value;
                markDirty();
            },
        });

        hostInput.update({
            .value = host,
            .hint = "Base URL (e.g. http://localhost:11434).",
            .submit = TextInput::Submit::OnEdit,
            .disabled = this->config.saving,
            .onChange = [this](const std::string& value) {
                host = value;
                markDirty();
            },
        });

        endpointControls.update({
            .items = {
                {.child = &nameInput},
                {.child = &hostInput},
            },
        });

        endpointField.update({
            .label = "Endpoint",
            .description = "Display name and base URL of the provider.",
            .control = &endpointControls,
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
        });

        pathsControls.update({
            .items = {
                {.child = &apiCombo},
                {.child = &chatPathInput},
                {.child = &modelsPathInput},
            },
        });

        pathsField.update({
            .label = "API",
            .description = "Request format and endpoints relative to the base URL.",
            .control = &pathsControls,
        });

        keyInput.update({
            .value = apiKey,
            .hint = this->config.editMode && !this->config.editProvider.apiKeyId.empty()
                        ? "Stored in the keychain. Type to replace."
                        : "API key",
            .submit = TextInput::Submit::OnEdit,
            .password = true,
            .disabled = this->config.saving,
            .onChange = [this](const std::string& value) {
                apiKey = value;
                markDirty();
            },
        });

        keyField.update({
            .label = "API Key",
            .description = "Saved in the system keychain, never in the settings file. Leave blank "
                           "for a custom endpoint without authentication.",
            .control = &keyInput,
            .divider = false,
        });

        const auto status = resolveStatus();
        statusText.update({
            .str = status.message,
            .tone = status.tone,
            .wrapped = true,
        });

        divider.update({});

        testButton.update({
            .str = this->config.testRunning ? "Testing..." : "Test Connection",
            .size = {Sakura::Fill, 40.0f},
            .disabled = !testEnabled(),
            .onClick = [this]() {
                if (!testEnabled()) {
                    return;
                }
                if (this->config.onTest) {
                    this->config.onTest(buildProvider(), apiKey, draftRevision);
                }
            },
        });

        addButton.update({
            .str = addLabel(),
            .size = {Sakura::Fill, 40.0f},
            .variant = Button::Variant::Action,
            .disabled = !addEnabled(),
            .onClick = [this]() {
                if (!addEnabled()) {
                    return;
                }
                if (this->config.onAdd) {
                    this->config.onAdd(buildProvider(), apiKey);
                }
            },
        });

        std::vector<Sakura::Stack::Item> items = {
            {.child = &header},
            {.child = &typeField},
            {.child = &endpointField},
            {.child = &pathsField},
            {.child = &keyField},
            {.child = &divider},
        };
        if (!status.message.empty()) {
            items.push_back({.child = &statusText});
        }
        items.push_back({.child = &testButton});
        items.push_back({.child = &addButton});
        stack.update({.items = std::move(items)});
        panel.update({
            .id = "IntelligenceProvider",
            .content = &stack,
            .backgroundColorKey = "popup_bg",
        });
    }

    Sakura::Component& component() {
        return panel;
    }

 private:
    struct Status {
        std::string message;
        Sakura::Text::Tone tone = Sakura::Text::Tone::Disabled;
    };

    void applyPreset(const std::string& label) {
        for (const auto& preset : config.presets) {
            if (preset.label == label) {
                applyProvider(preset.provider);
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
        provider.requestReasoning = requestReasoning;
        return provider;
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
        return config.isValid && config.isValid(buildProvider());
    }

    bool testEnabled() const {
        return !config.saving && !config.testRunning && providerValid();
    }

    bool addEnabled() const {
        return !config.saving && !config.testRunning && testPassed() && providerValid();
    }

    std::string addLabel() const {
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

    Status resolveStatus() const {
        if (config.testRunning) {
            return {"Testing connection...", Sakura::Text::Tone::Secondary};
        }
        if (testCurrent()) {
            return {config.testMessage,
                    config.testSuccess ? Sakura::Text::Tone::Success : Sakura::Text::Tone::Warning};
        }
        return {};
    }

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
    ModalHeader header;
    Sakura::Combo typeCombo;
    Sakura::SettingField typeField;
    Sakura::TextInput nameInput;
    Sakura::TextInput hostInput;
    Sakura::Stack endpointControls;
    Sakura::SettingField endpointField;
    Sakura::Combo apiCombo;
    Sakura::TextInput chatPathInput;
    Sakura::TextInput modelsPathInput;
    Sakura::Stack pathsControls;
    Sakura::SettingField pathsField;
    Sakura::TextInput keyInput;
    Sakura::SettingField keyField;
    Sakura::Divider divider;
    Sakura::Text statusText;
    Sakura::Button testButton;
    Sakura::Button addButton;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_INTELLIGENCE_PROVIDER_HH
