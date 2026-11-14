#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_UINT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_UINT_HH

#include "types.hh"

#include "../field_tooltip.hh"

namespace Jetstream {

struct FlowgraphConfigUIntField {
    using Config = FlowgraphConfigFieldConfig;

    Result update(Config config);
    Sakura::Component& component();

 private:
    Config config;
    Parser::Map parsedFormat;
    std::string unit;
    U64 value = 0;
    FlowgraphFieldTooltip help;
    Sakura::NodeField frame;
    Sakura::NodeUIntInput input;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_UINT_HH
