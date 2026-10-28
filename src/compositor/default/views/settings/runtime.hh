#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_RUNTIME_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_RUNTIME_HH

#include "../components/section_group.hh"
#include "components/settings_header.hh"
#include "../components/status_badge.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"
#include "jetstream/runtime_context_python.hh"

#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct RuntimeSettingsView {
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
        using Stack = Sakura::Stack;

        this->config = std::move(config);
        header.update({
            .title = "Runtime",
            .description = "Configure external runtime integrations.",
        });
        runtimeCombo.update({
            .id = "settings:python-runtime-selector",
            .options = runtimeOptions(),
            .value = selectedRuntimeOption(),
            .onChange = [this](const std::string& value) {
                handleRuntimeOptionChange(value);
            },
            .borderColorKey = "border",
        });
        pythonPathInput.update({
            .value = this->config.pythonPath,
            .hint = "/path/to/python",
            .submit = Sakura::TextInput::Submit::OnCommit,
            .onChange = [this](const std::string& value) {
                if (this->config.onPythonPathChange) {
                    this->config.onPythonPathChange(value);
                }
            },
            .borderColorKey = "border",
        });
        browseButton.update({
            .str = "Browse",
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
        pythonPathControls.update({
            .axis = Stack::Axis::Horizontal,
            .items = {
                {.child = &pythonPathInput, .size = 1.0f, .grow = 1.0f},
                {.child = &browseButton},
            },
        });
        pythonPathField.update({
            .content = &pythonPathControls,
            .border = false,
            .colorKey = "transparent",
            .hoveredColorKey = "transparent",
        });
        statusBadge.update({
            .label = validationBadge(),
            .tone = this->config.pythonValidation.valid ? StatusBadge::Tone::Success : StatusBadge::Tone::Error,
        });

        std::vector<Sakura::SettingCard::Row> pythonRows = {{
            .label = "Python Runtime",
            .description = "Choose one of the Python installations on your system, or a custom path.",
            .control = &runtimeCombo,
        }};
        if (selectedRuntimeOption() == CustomPathLabel()) {
            pythonRows.push_back({
                .label = "Custom Path",
                .description = "The Python interpreter used when no detected installation fits.",
                .control = &pythonPathField,
            });
        }
        if (!validationBadge().empty()) {
            pythonRows.push_back({
                .label = "Interpreter",
                .description = validationDescription(),
                .control = &statusBadge,
            });
        }
        pythonGroup.update({
            .title = "Python",
            .description = "The interpreter used to run Python blocks.",
            .rows = std::move(pythonRows),
        });

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
            .borderColorKey = "border",
        });
        dependencyGroup.update({
            .title = "Dependencies",
            .description = "How packages requested by Python blocks are installed.",
            .rows = {{
                .label = "Installation Policy",
                .description = "Choose how requested dependencies are installed.",
                .control = &dependencyPolicyCombo,
            }},
        });

        cacheButton.update({
            .str = "Open Folder",
            .onClick = this->config.onOpenPythonEnvironmentCache,
        });
        cacheGroup.update({
            .title = "Cache",
            .description = "Environments created for Python block dependencies.",
            .rows = {{
                .label = "Python Environment Cache",
                .description = "Browse cached Python dependency environments in your file manager.",
                .control = &cacheButton,
            }},
        });

        stack.update({
            .spacing = 4.0f,
            .items = {
                {.child = &header},
                {.child = &pythonGroup},
                {.child = &dependencyGroup},
                {.child = &cacheGroup},
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

        return config.pythonValidation.valid ? ICON_LC_CHECK "  Valid" : ICON_LC_X "  Invalid";
    }

    std::string validationDescription() const {
        if (!config.pythonValidation.valid) {
            return "The selected path is not a usable Python interpreter.";
        }
        return config.restartRequired ? "Restart CyberEther to apply the new interpreter."
                                      : "The selected interpreter is ready.";
    }

    Config config;
    SettingsHeader header;
    Sakura::Combo runtimeCombo;
    Sakura::TextInput pythonPathInput;
    Sakura::Button browseButton;
    Sakura::Stack pythonPathControls;
    Sakura::Div pythonPathField;
    StatusBadge statusBadge;
    SectionGroup pythonGroup;
    Sakura::Combo dependencyPolicyCombo;
    SectionGroup dependencyGroup;
    Sakura::Button cacheButton;
    SectionGroup cacheGroup;
    Sakura::Stack stack;
    bool customPathSelected = false;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_RUNTIME_HH
