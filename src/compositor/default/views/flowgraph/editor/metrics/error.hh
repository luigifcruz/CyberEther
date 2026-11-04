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
            .colorKey = "text_secondary",
            .align = Sakura::NodeLabel::Align::Center,
            .boxed = true,
        });
    }

    Sakura::Component& component() {
        return text;
    }

 private:
    Sakura::NodeLabel text;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_ERROR_HH
