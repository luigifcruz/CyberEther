#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_TYPES_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_TYPES_HH

#include "jetstream/render/sakura/sakura.hh"

#include <any>
#include <string>

namespace Jetstream {

struct FlowgraphMetricConfig {
    std::string id;
    std::string label;
    std::string help;
    std::string format;
    std::any value;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_TYPES_HH
