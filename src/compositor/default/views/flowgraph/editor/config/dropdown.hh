#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_DROPDOWN_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_DROPDOWN_HH

#include "types.hh"

#include "../field_tooltip.hh"

namespace Jetstream {

struct FlowgraphConfigDropdownField {
    using Config = FlowgraphConfigFieldConfig;

    Result update(Config config);
    Sakura::Component& component();

 private:
    std::string currentLabel() const;
    void parseFormat();

    Config config;
    Parser::Map parsedFormat;
    std::vector<std::string> values;
    std::vector<std::string> labels;
    int currentIndex = -1;
    FlowgraphFieldTooltip help;
    Sakura::NodeField frame;
    Sakura::NodeCombo combo;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_DROPDOWN_HH
