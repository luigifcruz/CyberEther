#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_TABLE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_TABLE_HH

#include "error.hh"
#include "types.hh"

#include "../field_tooltip.hh"

namespace Jetstream {

struct FlowgraphMetricTable {
    using Config = FlowgraphMetricConfig;

    void update(Config config);
    Sakura::Component& component();

 private:
    void parseValue();

    Config config;
    std::optional<Parser::Map> parsedValue;
    std::vector<std::string> columns;
    std::vector<std::vector<std::string>> rows;
    std::string errorText;
    FlowgraphFieldTooltip help;
    Sakura::NodeField frame;
    FlowgraphMetricError error;
    Sakura::NodeTable table;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_TABLE_HH
