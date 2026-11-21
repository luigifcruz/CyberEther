#include "progressbar.hh"

namespace Jetstream {

void FlowgraphMetricProgressBar::update(Config config) {
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
        .colorKey = colorKey,
    });
    error.update({
        .id = this->config.id + "Error",
        .str = errorText,
    });
}

Sakura::Component& FlowgraphMetricProgressBar::component() {
    return frame;
}

void FlowgraphMetricProgressBar::parseValue() {
    errorText.clear();
    overlay.clear();
    value = 0.0f;
    colorKey = "button_active";
    if (!config.value.has_value()) {
        errorText = "No metric";
        return;
    }

    try {
        const auto metric = std::any_cast<std::pair<std::string, F32>>(config.value);
        overlay = metric.first;
        if (!std::isfinite(metric.second)) {
            return;
        }
        value = std::clamp(metric.second, 0.0f, 1.0f);
        colorKey = levelColorKey(metric.second);
    } catch (const std::bad_any_cast&) {
        errorText = "Invalid metric type";
    }
}

std::string FlowgraphMetricProgressBar::levelColorKey(const F32 level) const {
    const auto past = [&](const char* key, const bool above) {
        F32 threshold = 0.0f;
        return config.format.contains(key) &&
               Parser::Deserialize(config.format, key, threshold) == Result::SUCCESS &&
               (above ? level > threshold : level < threshold);
    };
    if (past("error_above", true) || past("error_below", false)) {
        return "progress_error";
    }
    if (past("warning_above", true) || past("warning_below", false)) {
        return "progress_warning";
    }
    return "button_active";
}

}  // namespace Jetstream
