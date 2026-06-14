#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_RUNTIME_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_RUNTIME_HH

#include "jetstream/render/sakura/sakura.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"
#include "jetstream/runtime_context_python.hh"

#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct RuntimeSettingsPanel : public Sakura::Component {
    struct Config {
        std::string pythonPath;
        std::vector<PythonRuntimeContext::Candidate> pythonCandidates;
        PythonRuntimeContext::Validation pythonValidation;
        bool restartRequired = false;
        std::function<void(const std::string&)> onPythonPathChange;
        std::function<void(const std::string&, std::function<void(std::string)>)> onBrowsePythonPath;
    };

    void update(Config config) {
        this->config = std::move(config);

        title.update({
            .id = "RuntimeTitle",
            .str = "Runtime",
            .font = Sakura::Text::Font::Bold,
            .scale = 1.2f,
        });

        description.update({
            .id = "RuntimeDescription",
            .str = "Configure external runtime integrations.",
            .tone = Sakura::Text::Tone::Secondary,
            .wrapped = true,
        });

        divider.update({
            .id = "RuntimeHeaderDivider",
        });

        runtimeField.update({
            .id = "PythonRuntimeSelectorField",
        });

        labelRow.update({
            .id = "PythonRuntimeLabelRow",
            .spacing = 4.0f,
        });

        labelText.update({
            .id = "PythonRuntimeLabelText",
            .str = "Python Runtime",
        });

        runtimeDescription.update({
            .id = "PythonRuntimeDescription",
            .str = "Choose one of the Python installations found on your system, or a custom path.",
            .tone = Sakura::Text::Tone::Disabled,
            .wrapped = true,
        });

        runtimeCombo.update({
            .id = "##app-settings-python-runtime-selector",
            .options = runtimeOptions(),
            .value = selectedRuntimeOption(),
            .onChange = [this](const std::string& value) {
                handleRuntimeOptionChange(value);
            },
        });

        pythonPathInput.update({
            .id = "##app-settings-python-runtime-path",
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
            .id = "PythonRuntimeBrowse",
            .str = "Browse File",
            .size = {-1.0f, 0.0f},
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
            .id = "PythonRuntimeStatusText",
            .str = validationBadge(),
            .colorKey = validationBadgeColorKey(),
        });
    }

    void render(const Sakura::Context& ctx) const {
        title.render(ctx);
        description.render(ctx);
        divider.render(ctx);

        runtimeField.render(ctx, [&](const Sakura::Context& ctx) {
            Sakura::HStack::Children labelChildren;
            labelChildren.push_back([this](const Sakura::Context& ctx) { labelText.render(ctx); });
            if (!validationBadge().empty()) {
                labelChildren.push_back([this](const Sakura::Context& ctx) { statusText.render(ctx); });
            }
            labelRow.render(ctx, std::move(labelChildren));

            runtimeCombo.render(ctx);

            if (selectedRuntimeOption() == CustomPathLabel()) {
                pythonPathInput.render(ctx);
                browseButton.render(ctx);
            }

            runtimeDescription.render(ctx);
        });

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
    Sakura::SettingField runtimeField;
    Sakura::HStack labelRow;
    Sakura::Text labelText;
    Sakura::Text runtimeDescription;
    Sakura::Combo runtimeCombo;
    Sakura::TextInput pythonPathInput;
    Sakura::Button browseButton;
    Sakura::Text statusText;
    bool customPathSelected = false;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_RUNTIME_HH
