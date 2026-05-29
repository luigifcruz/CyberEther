#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_DEVELOPER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_DEVELOPER_HH

#include "jetstream/render/sakura/sakura.hh"

#include "jetstream/logger.hh"

#include <functional>
#include <string>

namespace Jetstream {

struct DeveloperSettingsPanel : public Sakura::Component {
    struct Config {
        bool latencyEnabled = false;
        bool timingEnabled = false;
        int logLevel = JST_LOG_DEBUG_DEFAULT_LEVEL;
        std::function<void(bool)> onLatencyEnabledChange;
        std::function<void(bool)> onTimingEnabledChange;
        std::function<void(int)> onLogLevelChange;
    };

    void update(Config config) {
        this->config = std::move(config);

        title.update({
            .id = "DeveloperTitle",
            .str = "Developer",
            .font = Sakura::Text::Font::Bold,
            .scale = 1.2f,
        });

        description.update({
            .id = "DeveloperDescription",
            .str = "Diagnostics and development tools.",
            .tone = Sakura::Text::Tone::Secondary,
            .wrapped = true,
        });

        divider.update({
            .id = "DeveloperHeaderDivider",
        });

        latencyField.update({
            .id = "DeveloperLatencyField",
            .description = "Display real-time signal latency measurements.",
        });

        latencyCheckbox.update({
            .id = "DeveloperLatencyCheckbox",
            .label = "Show latency window",
            .value = this->config.latencyEnabled,
            .onChange = [this](bool value) {
                if (this->config.onLatencyEnabledChange) {
                    this->config.onLatencyEnabledChange(value);
                }
            },
        });

        timingField.update({
            .id = "DeveloperTimingField",
            .description = "Display engine timing information.",
        });

        timingCheckbox.update({
            .id = "DeveloperTimingCheckbox",
            .label = "Show timing",
            .value = this->config.timingEnabled,
            .onChange = [this](bool value) {
                if (this->config.onTimingEnabledChange) {
                    this->config.onTimingEnabledChange(value);
                }
            },
        });

        logLevelField.update({
            .id = "DeveloperLogLevelField",
            .label = "Log Level",
            .description = "Set the verbosity level for console and file log output.",
            .divider = false,
        });

        logLevelCombo.update({
            .id = "##app-settings-log-level",
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
    }

    void render(const Sakura::Context& ctx) const {
        title.render(ctx);
        description.render(ctx);
        divider.render(ctx);

        latencyField.render(ctx, [&](const Sakura::Context& ctx) {
            latencyCheckbox.render(ctx);
        });

        timingField.render(ctx, [&](const Sakura::Context& ctx) {
            timingCheckbox.render(ctx);
        });

        logLevelField.render(ctx, [&](const Sakura::Context& ctx) {
            logLevelCombo.render(ctx);
        });
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
    Sakura::SettingField latencyField;
    Sakura::SettingField timingField;
    Sakura::SettingField logLevelField;
    Sakura::Checkbox latencyCheckbox;
    Sakura::Checkbox timingCheckbox;
    Sakura::Combo logLevelCombo;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_DEVELOPER_HH
