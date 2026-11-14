#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_CONSOLE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_CONSOLE_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphConfigPythonConsoleField {
    using Config = FlowgraphConfigFieldConfig;

    FlowgraphNodeHeightSpec heightSpec() const;
    Result update(Config config);
    Sakura::Component& component();

 private:
    Config config;
    bool loaded = false;
    Sakura::NodeField frame;
    Sakura::Console console;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_CONSOLE_HH
