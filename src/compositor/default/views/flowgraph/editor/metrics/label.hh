#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_LABEL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_LABEL_HH

#include "error.hh"
#include "types.hh"

#include "../field_tooltip.hh"

namespace Jetstream {

struct FlowgraphMetricLabel {
    using Config = FlowgraphMetricConfig;

    void update(Config config);
    Sakura::Component& component();

 private:
    void parseValue();

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
