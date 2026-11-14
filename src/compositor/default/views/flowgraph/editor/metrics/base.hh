#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_BASE_HH

#include "error.hh"
#include "label.hh"
#include "progressbar.hh"
#include "table.hh"
#include "types.hh"

#include "../field_tooltip.hh"

#include "jetstream/parser.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphMetricInstance {
    void update(FlowgraphMetricConfig config);
    Sakura::Component& component();

 private:
    std::string kind;
    FlowgraphMetricProgressBar progress;
    FlowgraphMetricLabel label;
    FlowgraphMetricTable table;
    FlowgraphFieldTooltip unknownHelp;
    Sakura::NodeField unknownFrame;
    FlowgraphMetricError unknownText;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_BASE_HH
