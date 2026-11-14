#include "developer.hh"

namespace Jetstream {

void DeveloperSettingsView::update(Config config) {
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
    logLevelStepper.update({
        .options = {"Error", "Warning", "Info", "Debug", "Trace"},
        .selected = static_cast<U64>(std::clamp(this->config.logLevel, 0, 4)),
        .fallback = static_cast<U64>(JST_LOG_DEBUG_DEFAULT_LEVEL),
        .onSelect = [this](U64 index) {
            if (this->config.onLogLevelChange) {
                this->config.onLogLevelChange(static_cast<int>(index));
            }
        },
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
                .control = &logLevelStepper,
            },
        },
    });
    benchmarksButton.update({
        .str = "Open",
        .icon = ICON_LC_GAUGE,
        .onClick = [this]() {
            if (this->config.onShowBenchmarks) {
                this->config.onShowBenchmarks();
            }
        },
    });
    reloadButton.update({
        .str = "Reload",
        .icon = ICON_LC_REFRESH_CW,
        .onClick = [this]() {
            if (this->config.onReloadPlugins) {
                this->config.onReloadPlugins();
            }
        },
    });
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
        .spacing = Sakura::Spacing::XSmall,
        .items = {
            {.child = &header},
            {.child = &diagnosticsGroup},
            {.child = &toolsGroup},
        },
    });
}

Sakura::Component& DeveloperSettingsView::content() {
    return stack;
}

}  // namespace Jetstream
