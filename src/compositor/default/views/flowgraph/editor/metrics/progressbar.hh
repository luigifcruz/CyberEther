#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_PROGRESSBAR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_PROGRESSBAR_HH

#include "error.hh"
#include "types.hh"

#include "../field_tooltip.hh"

#include <algorithm>

namespace Jetstream {

struct FlowgraphMetricProgressBar {
    using Config = FlowgraphMetricConfig;

    void update(Config config) {
        this->config = std::move(config);
        help.update(this->config.id, this->config.help);
        parseValue();
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .tooltip = help.overlay(),
            .background = false,
            .content = errorText.empty() ? static_cast<Sakura::Component*>(&progress) : &error.component(),
        });
        progress.update({
            .id = this->config.id + "Progress",
            .value = value,
            .overlay = overlay,
        });
        error.update({
            .id = this->config.id + "Error",
            .str = errorText,
        });
    }

    Sakura::Component& component() {
        return frame;
    }

 private:
    void parseValue() {
        errorText.clear();
        overlay.clear();
        value = 0.0f;
        if (!config.value.has_value()) {
            errorText = "No metric";
            return;
        }

        try {
            const auto metric = std::any_cast<std::pair<std::string, F32>>(config.value);
            overlay = metric.first;
            value = std::clamp(metric.second, 0.0f, 1.0f);
        } catch (const std::bad_any_cast&) {
            errorText = "Invalid metric type";
        }
    }

    Config config;
    std::string overlay;
    F32 value = 0.0f;
    std::string errorText;
    FlowgraphFieldTooltip help;
    Sakura::NodeField frame;
    FlowgraphMetricError error;
    Sakura::NodeProgressBar progress;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_PROGRESSBAR_HH
