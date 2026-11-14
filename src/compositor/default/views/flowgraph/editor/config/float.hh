#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_FLOAT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_FLOAT_HH

#include "types.hh"

#include "../field_tooltip.hh"

namespace Jetstream {

struct FlowgraphConfigFloatField {
    using Config = FlowgraphConfigFieldConfig;

    Result update(Config config);
    Sakura::Component& component();

 private:
    void parseFormat();

    Config config;
    Parser::Map parsedFormat;
    F32 multiplier = 1.0f;
    std::string unit;
    std::string stepConfig;
    int precision = 2;
    F32 value = 0.0f;
    F32 step = 0.0f;
    bool hasStep = false;
    FlowgraphFieldTooltip help;
    Sakura::NodeField frame;
    Sakura::NodeFloatInput input;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_FLOAT_HH
