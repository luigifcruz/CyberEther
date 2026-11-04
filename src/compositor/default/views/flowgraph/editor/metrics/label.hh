#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_LABEL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_LABEL_HH

#include "error.hh"
#include "types.hh"

#include "../field_tooltip.hh"

namespace Jetstream {

struct FlowgraphMetricLabel {
    using Config = FlowgraphMetricConfig;

    void update(Config config) {
        this->config = std::move(config);
        parseValue();
        help.update(this->config.id, this->config.help);
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .tooltip = help.overlay(),
            .background = false,
            .content = errorText.empty() ? static_cast<Sakura::Component*>(&text) : &error.component(),
        });
        text.update({
            .id = this->config.id + "Text",
            .str = value,
            .align = Sakura::NodeLabel::Align::Right,
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
        value.clear();
        errorText.clear();
        if (!config.value.has_value()) {
            errorText = "No metric";
            return;
        }

        try {
            value = std::any_cast<std::string>(config.value);
        } catch (const std::bad_any_cast&) {
            errorText = "Invalid metric type";
        }
    }

    Config config;
    std::string value;
    std::string errorText;
    FlowgraphFieldTooltip help;
    Sakura::NodeField frame;
    FlowgraphMetricError error;
    Sakura::NodeLabel text;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_LABEL_HH
