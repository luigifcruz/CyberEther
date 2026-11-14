#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_BOOL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_BOOL_HH

#include "types.hh"

#include "../field_tooltip.hh"

namespace Jetstream {

struct FlowgraphConfigBoolField {
    using Config = FlowgraphConfigFieldConfig;

    Result update(Config config);
    Sakura::Component& component();

 private:
    Config config;
    bool value = false;
    FlowgraphFieldTooltip help;
    Sakura::NodeField frame;
    Sakura::NodeBoolInput input;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_BOOL_HH
