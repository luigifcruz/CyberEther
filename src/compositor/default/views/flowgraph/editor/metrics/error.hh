#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_ERROR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_ERROR_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphMetricError {
    struct Config {
        std::string id;
        std::string str;
    };

    void update(Config config);
    Sakura::Component& component();

 private:
    Sakura::NodeLabel text;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_ERROR_HH
