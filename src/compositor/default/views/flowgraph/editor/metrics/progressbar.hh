#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_PROGRESSBAR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_PROGRESSBAR_HH

#include "error.hh"
#include "types.hh"

#include "../field_tooltip.hh"

#include <algorithm>
#include <cmath>

namespace Jetstream {

struct FlowgraphMetricProgressBar {
    using Config = FlowgraphMetricConfig;

    void update(Config config);
    Sakura::Component& component();

 private:
    void parseValue();
    std::string levelColorKey(const F32 level) const;

    Config config;
    std::string overlay;
    F32 value = 0.0f;
    std::string colorKey = "button_active";
    std::string errorText;
    FlowgraphFieldTooltip help;
    Sakura::NodeField frame;
    FlowgraphMetricError error;
    Sakura::NodeProgressBar progress;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_PROGRESSBAR_HH
