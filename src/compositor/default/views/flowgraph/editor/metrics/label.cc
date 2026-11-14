#include "label.hh"

namespace Jetstream {

void FlowgraphMetricLabel::update(Config config) {
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

Sakura::Component& FlowgraphMetricLabel::component() {
    return frame;
}

void FlowgraphMetricLabel::parseValue() {
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

}  // namespace Jetstream
