#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_DEVELOPER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_DEVELOPER_HH

#include "components/settings_group.hh"
#include "components/settings_header.hh"

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
        this->config = std::move(config);
        header.update({
            .title = "Developer",
            .description = "Diagnostics and development tools.",
        });
        timingToggle.update({
            .value = this->config.timingEnabled,
            .onChange = [this](bool value) {
                if (this->config.onTimingEnabledChange) {
                    this->config.onTimingEnabledChange(value);
                }
            },
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
            .borderColorKey = "border",
        });
        diagnosticsGroup.update({
            .title = "Diagnostics",
            .description = "Logging and timing information.",
            .rows = {
                {
                    .label = "Show Timing",
                    .description = "Display engine timing information.",
                    .control = &timingToggle,
                },
                {
                    .label = "Log Level",
                    .description = "Verbosity of console and file log output.",
                    .control = &logLevelCombo,
                },
            },
        });
        benchmarksButton.update(compactButton("Open", [this]() {
            if (this->config.onShowBenchmarks) {
                this->config.onShowBenchmarks();
            }
        }));
        reloadButton.update(compactButton("Reload", [this]() {
            if (this->config.onReloadPlugins) {
                this->config.onReloadPlugins();
            }
        }));
        toolsGroup.update({
            .title = "Tools",
            .description = "Utilities for working on blocks and plugins.",
            .rows = {
                {
                    .label = "Benchmarks",
                    .description = "Measure the throughput of every block on the available devices.",
                    .control = &benchmarksButton,
                },
                {
                    .label = "Plugins",
                    .description = "Reload every registered plugin from disk.",
                    .control = &reloadButton,
                },
            },
        });
        stack.update({
            .spacing = 4.0f,
            .items = {
                {.child = &header},
                {.child = &diagnosticsGroup},
                {.child = &toolsGroup},
            },
        });
    }

    Sakura::Component& content() {
        return stack;
    }

 private:
    static Sakura::Button::Config compactButton(std::string label, std::function<void()> onClick) {
        return {
            .str = std::move(label),
            .textScale = 0.9f,
            .padding = {9.0f, 5.0f},
            .onClick = std::move(onClick),
            .font = Sakura::Text::Font::Sans,
        };
    }

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
    SettingsHeader header;
    Sakura::Toggle timingToggle;
    Sakura::Combo logLevelCombo;
    SettingsGroup diagnosticsGroup;
    Sakura::Button benchmarksButton;
    Sakura::Button reloadButton;
    SettingsGroup toolsGroup;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_DEVELOPER_HH
