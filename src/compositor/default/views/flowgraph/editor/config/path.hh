#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_PATH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_PATH_HH

#include "types.hh"

#include "../field_tooltip.hh"

namespace Jetstream {

struct FlowgraphConfigPathField {
    using Config = FlowgraphConfigFieldConfig;

    enum class Mode {
        Open,
        Save,
    };

    Result update(Config config);
    Sakura::Component& component();

 private:
    void parseFormat();
    void browse() const;

    Config config;
    Parser::Map parsedFormat;
    std::vector<std::string> extensions;
    std::string value;
    Mode mode = Mode::Open;
    FlowgraphFieldTooltip help;
    FlowgraphFieldTooltip pathTooltip;
    Sakura::NodeField frame;
    Sakura::NodePathInput input;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_PATH_HH
