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
    bool wide(const std::string& name) const;
    std::any patchValue(const std::string& name, F64 next) const;

    Config config;
    Parser::Map parsedFormat;
    F64 multiplier = 1.0;
    std::string unit;
    std::string stepConfig;
    int precision = 2;
    F64 value = 0.0;
    F64 step = 0.0;
    bool hasStep = false;
    FlowgraphFieldTooltip help;
    Sakura::NodeField frame;
    Sakura::NodeFloatInput input;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_FLOAT_HH
