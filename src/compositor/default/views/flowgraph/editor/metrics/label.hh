#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_LABEL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_LABEL_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphMetricLabel : public Sakura::Component {
    using Config = FlowgraphMetricConfig;

    void update(Config config) {
        this->config = std::move(config);
        parseValue();
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .help = this->config.help,
            .background = false,
        });
        text.update({
            .id = this->config.id + "Text",
            .str = value,
            .align = Sakura::Text::Align::Right,
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
                text.render(ctx);
            }
        });
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
    Sakura::NodeField frame;
    Sakura::NodeLabel error;
    Sakura::NodeLabel text;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_LABEL_HH
