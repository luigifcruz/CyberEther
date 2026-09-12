#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_ERROR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_ERROR_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphMetricError {
    struct Config {
        std::string id;
        std::string str;
    };

    void update(Config config) {
        text.update({
            .id = config.id,
            .str = std::move(config.str),
            .tone = Sakura::Text::Tone::Secondary,
            .align = Sakura::Text::Align::Center,
            .clipped = true,
            .boxed = true,
        });
    }

    void render(const Sakura::Context& ctx) const {
        text.render(ctx);
    }

 private:
    Sakura::NodeLabel text;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_ERROR_HH
