#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_RUNTIME_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_RUNTIME_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"
#include "jetstream/runtime_context_python.hh"

#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct RuntimeSettingsPanel {
    struct Config {
        std::string pythonPath;
        std::string dependencyPolicy = "prompt";
        std::vector<PythonRuntimeContext::Candidate> pythonCandidates;
        PythonRuntimeContext::Validation pythonValidation;
        bool restartRequired = false;
        std::function<void(const std::string&)> onPythonPathChange;
        std::function<void(const std::string&)> onDependencyPolicyChange;
        std::function<void()> onOpenPythonEnvironmentCache;
        std::function<void(const std::string&, std::function<void(std::string)>)> onBrowsePythonPath;
    };

    void update(Config config) {
        using Text = Sakura::Text;
        using Stack = Sakura::Stack;

        this->config = std::move(config);
        title.update({
            .str = "Runtime",
            .font = Text::Font::Bold,
            .scale = 1.2f,
        });
        description.update({
            .str = "Configure external runtime integrations.",
            .tone = Text::Tone::Secondary,
            .wrapped = true,
        });
        divider.update({});
        labelText.update({.str = "Python Runtime"});
        runtimeDescription.update({
            .str = "Choose one of the Python installations found on your system, or a custom path.",
            .tone = Text::Tone::Disabled,
            .wrapped = true,
        });
        runtimeCombo.update({
            .id = "settings:python-runtime-selector",
            .options = runtimeOptions(),
            .value = selectedRuntimeOption(),
            .onChange = [this](const std::string& value) {
                handleRuntimeOptionChange(value);
            },
        });
        pythonPathInput.update({
            .value = this->config.pythonPath,
            .hint = "Path to Python (e.g. /path/to/python).",
            .submit = Sakura::TextInput::Submit::OnCommit,
            .onChange = [this](const std::string& value) {
                if (this->config.onPythonPathChange) {
                    this->config.onPythonPathChange(value);
                }
            },
        });
        browseButton.update({
            .str = "Browse File",
            .size = {Sakura::Fill, 0.0f},
            .onClick = [this]() {
                if (this->config.onBrowsePythonPath) {
                    this->config.onBrowsePythonPath(this->config.pythonPath, [this](std::string path) {
                        if (this->config.onPythonPathChange) {
                            this->config.onPythonPathChange(std::move(path));
                        }
                    });
                }
            },
        });
        statusText.update({
            .str = validationBadge(),
            .colorKey = validationBadgeColorKey(),
        });

        std::vector<Stack::Item> labelItems = {{.child = &labelText}};
        if (!validationBadge().empty()) {
            labelItems.push_back({.child = &statusText});
        }
        labelRow.update({
            .axis = Stack::Axis::Horizontal,
            .spacing = 12.0f,
            .items = std::move(labelItems),
        });
        std::vector<Stack::Item> runtimeItems = {{.child = &labelRow}, {.child = &runtimeCombo}};
        if (selectedRuntimeOption() == CustomPathLabel()) {
            runtimeItems.push_back({.child = &pythonPathInput});
            runtimeItems.push_back({.child = &browseButton});
        }
        runtimeItems.push_back({.child = &runtimeDescription});
        runtimeControls.update({.items = std::move(runtimeItems)});
        runtimeField.update({.control = &runtimeControls});

        dependencyPolicyCombo.update({
            .id = "settings:runtime-dependency-policy",
            .options = {
                "Ask Every Time",
                "Install Automatically",
                "Never Install",
            },
            .value = dependencyPolicyLabel(this->config.dependencyPolicy),
            .onChange = [this](const std::string& label) {
                if (this->config.onDependencyPolicyChange) {
                    this->config.onDependencyPolicyChange(dependencyPolicyValue(label));
                }
            },
        });
        dependencyPolicyField.update({
            .label = "Dependency Installation Policy",
            .description = "Choose how requested dependencies are installed.",
            .control = &dependencyPolicyCombo,
        });
        cacheTitle.update({.str = "Python Environment Cache"});
        cacheButton.update({
            .str = "Open Cache Folder",
            .size = {Sakura::Fill, 34.0f},
            .onClick = this->config.onOpenPythonEnvironmentCache,
        });
        cacheDescription.update({
            .str = "Browse cached Python dependency environments in your file manager.",
            .tone = Text::Tone::Secondary,
            .wrapped = true,
        });
        cacheFooterDivider.update({});
        stack.update({
            .items = {
                {.child = &title},
                {.child = &description},
                {.child = &divider},
                {.child = &dependencyPolicyField},
                {.child = &runtimeField},
                {.child = &cacheTitle},
                {.child = &cacheButton},
                {.child = &cacheDescription},
                {.child = &cacheFooterDivider},
            },
        });
    }

    Sakura::Component& content() {
        return stack;
    }

 private:
    static const std::string& AutoRuntimeLabel() {
        static const std::string label = "Auto";
        return label;
    }

    static const std::string& CustomPathLabel() {
        static const std::string label = "Custom Path";
        return label;
    }

    static std::string dependencyPolicyLabel(const std::string& value) {
        if (value == "allow") {
            return "Install Automatically";
        }
        if (value == "deny") {
            return "Never Install";
        }
        return "Ask Every Time";
    }

    static std::string dependencyPolicyValue(const std::string& label) {
        if (label == "Install Automatically") {
            return "allow";
        }
        if (label == "Never Install") {
            return "deny";
        }
        return "prompt";
    }

    std::vector<std::string> runtimeOptions() const {
        std::vector<std::string> options;
        options.reserve(config.pythonCandidates.size() + 2);
        options.push_back(AutoRuntimeLabel());
        for (const auto& candidate : config.pythonCandidates) {
            options.push_back(candidate.label);
        }
        options.push_back(CustomPathLabel());
        return options;
    }

    std::string selectedRuntimeOption() const {
        if (customPathSelected) {
            return CustomPathLabel();
        }

        if (config.pythonPath.empty()) {
            return AutoRuntimeLabel();
        }

        const auto match = std::find_if(config.pythonCandidates.begin(),
                                        config.pythonCandidates.end(),
                                        [this](const auto& candidate) {
                                            return candidate.path == config.pythonPath;
                                        });
        if (match != config.pythonCandidates.end()) {
            return match->label;
        }

        return CustomPathLabel();
    }

    void handleRuntimeOptionChange(const std::string& value) {
        if (!config.onPythonPathChange) {
            return;
        }

        if (value == AutoRuntimeLabel()) {
            customPathSelected = false;
            config.onPythonPathChange("");
            return;
        }

        if (value == CustomPathLabel()) {
            customPathSelected = true;
            return;
        }

        customPathSelected = false;

        const auto match = std::find_if(config.pythonCandidates.begin(),
                                        config.pythonCandidates.end(),
                                        [&](const auto& candidate) {
                                            return candidate.label == value;
                                        });
        if (match != config.pythonCandidates.end()) {
            config.onPythonPathChange(match->path);
        }
    }

    std::string validationBadge() const {
        if (selectedRuntimeOption() == CustomPathLabel() && config.pythonPath.empty()) {
            return "";
        }

        if (!config.pythonValidation.valid) {
            return ICON_FA_XMARK " Invalid";
        }

        return config.restartRequired
            ? ICON_FA_CHECK " Valid File (restart CyberEther to apply)"
            : ICON_FA_CHECK " Valid File";
    }

    std::string validationBadgeColorKey() const {
        return config.pythonValidation.valid ? "success_green" : "error_red";
    }

    Config config;
    Sakura::Text title;
    Sakura::Text description;
    Sakura::Divider divider;
    Sakura::Text labelText;
    Sakura::Text statusText;
    Sakura::Stack labelRow;
    Sakura::Combo runtimeCombo;
    Sakura::TextInput pythonPathInput;
    Sakura::Button browseButton;
    Sakura::Text runtimeDescription;
    Sakura::Stack runtimeControls;
    Sakura::SettingField runtimeField;
    Sakura::Combo dependencyPolicyCombo;
    Sakura::SettingField dependencyPolicyField;
    Sakura::Text cacheTitle;
    Sakura::Button cacheButton;
    Sakura::Text cacheDescription;
    Sakura::Divider cacheFooterDivider;
    Sakura::Stack stack;
    bool customPathSelected = false;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_RUNTIME_HH
