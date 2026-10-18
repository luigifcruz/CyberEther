#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_DEVELOPER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_DEVELOPER_HH

#include "jetstream/render/sakura/base.hh"

#include "jetstream/logger.hh"

#include <functional>
#include <string>

namespace Jetstream {

struct DeveloperSettingsPanel {
    struct Config {
        bool timingEnabled = false;
        int logLevel = JST_LOG_DEBUG_DEFAULT_LEVEL;
        std::function<void(bool)> onTimingEnabledChange;
        std::function<void(int)> onLogLevelChange;
        std::function<void()> onShowBenchmarks;
        std::function<void()> onReloadPlugins;
    };

    void update(Config config) {
        using Text = Sakura::Text;

        this->config = std::move(config);
        title.update({
            .str = "Developer",
            .font = Text::Font::Bold,
            .scale = 1.2f,
        });
        description.update({
            .str = "Diagnostics and development tools.",
            .tone = Text::Tone::Secondary,
            .wrapped = true,
        });
        divider.update({});
        timingCheckbox.update({
            .label = "Show timing",
            .value = this->config.timingEnabled,
            .onChange = [this](bool value) {
                if (this->config.onTimingEnabledChange) {
                    this->config.onTimingEnabledChange(value);
                }
            },
        });
        timingField.update({
            .description = "Display engine timing information.",
            .control = &timingCheckbox,
        });
        logLevelCombo.update({
            .id = "settings:log-level",
            .options = {
                "Fatal / Error",
                "Warnings",
                "Info",
                "Debug",
                "Trace",
            },
            .value = logLevelLabel(this->config.logLevel),
            .onChange = [this](const std::string& label) {
                const int value = logLevelValue(label);
                if (this->config.onLogLevelChange) {
                    this->config.onLogLevelChange(value);
                }
            },
        });
        logLevelField.update({
            .label = "Log Level",
            .description = "Set the verbosity level for console and file log output.",
            .control = &logLevelCombo,
        });
        benchmarksButton.update({
            .str = "Open Benchmarks",
            .size = {Sakura::Fill, 0.0f},
            .onClick = [this]() {
                if (this->config.onShowBenchmarks) {
                    this->config.onShowBenchmarks();
                }
            },
        });
        benchmarksField.update({
            .label = "Benchmarks",
            .description = "Measure the throughput of every block on the available devices.",
            .control = &benchmarksButton,
        });
        reloadButton.update({
            .str = "Reload Plugins",
            .size = {Sakura::Fill, 0.0f},
            .onClick = [this]() {
                if (this->config.onReloadPlugins) {
                    this->config.onReloadPlugins();
                }
            },
        });
        reloadField.update({
            .label = "Plugins",
            .description = "Reload every registered plugin from disk.",
            .control = &reloadButton,
            .divider = false,
        });
        stack.update({
            .items = {
                {.child = &title},
                {.child = &description},
                {.child = &divider},
                {.child = &timingField},
                {.child = &logLevelField},
                {.child = &benchmarksField},
                {.child = &reloadField},
            },
        });
    }

    Sakura::Component& content() {
        return stack;
    }

 private:
    static std::string logLevelLabel(int logLevel) {
        switch (logLevel) {
            case 0: return "Fatal / Error";
            case 1: return "Warnings";
            case 2: return "Info";
            case 3: return "Debug";
            case 4: return "Trace";
        }
        return "Unknown";
    }

    static int logLevelValue(const std::string& label) {
        if (label == "Fatal / Error") return 0;
        if (label == "Warnings") return 1;
        if (label == "Info") return 2;
        if (label == "Debug") return 3;
        if (label == "Trace") return 4;
        return JST_LOG_DEBUG_DEFAULT_LEVEL;
    }

    Config config;
    Sakura::Text title;
    Sakura::Text description;
    Sakura::Divider divider;
    Sakura::Checkbox timingCheckbox;
    Sakura::Combo logLevelCombo;
    Sakura::SettingField timingField;
    Sakura::SettingField logLevelField;
    Sakura::Button benchmarksButton;
    Sakura::Button reloadButton;
    Sakura::SettingField benchmarksField;
    Sakura::SettingField reloadField;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_DEVELOPER_HH
