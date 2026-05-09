#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_PROGRESSBAR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_PROGRESSBAR_HH

#include "types.hh"

#include <algorithm>

namespace Jetstream {

struct FlowgraphMetricProgressBar : public Sakura::Component {
    using Config = FlowgraphMetricConfig;

    void update(Config config) {
        this->config = std::move(config);
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .help = this->config.help,
            .background = false,
        });
        parseValue();
        progress.update({
            .id = this->config.id + "Progress",
            .value = value,
            .overlay = overlay,
            .size = {-1.0f, 0.0f},
        });
        error.update({
            .id = this->config.id + "Error",
            .str = errorText,
            .tone = Sakura::Text::Tone::Warning,
        });
    }

    void render(const Sakura::Context& ctx) const {
        frame.render(ctx, [this](const Sakura::Context& ctx) {
            if (!errorText.empty()) {
                error.render(ctx);
            } else {
                progress.render(ctx);
            }
        });
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
    Sakura::NodeField frame;
    Sakura::NodeLabel error;
    Sakura::NodeProgressBar progress;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_PROGRESSBAR_HH
