#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MARKDOWN_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MARKDOWN_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphConfigMarkdownField {
    using Config = FlowgraphConfigFieldConfig;

    FlowgraphNodeHeightSpec heightSpec() const;
    Result update(Config config);
    Sakura::Component& component();

 private:
    Config config;
    std::string value;
    std::optional<std::string> typed;
    bool editing = false;
    Sakura::NodeField frame;
    Sakura::MarkdownEditor markdownField;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MARKDOWN_HH
