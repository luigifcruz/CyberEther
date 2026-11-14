#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_PYTHON_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_PYTHON_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphConfigPythonField {
    using Config = FlowgraphConfigFieldConfig;

    FlowgraphNodeHeightSpec heightSpec() const;
    Result update(Config config);
    Sakura::Component& component();

 private:
    Config config;
    std::string parsedEncoded;
    Sakura::NodeField frame;
    Sakura::CodeEditor editor;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_PYTHON_HH
