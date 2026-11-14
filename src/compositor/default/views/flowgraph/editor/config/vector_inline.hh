#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_INLINE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_INLINE_HH

#include "types.hh"

#include "../field_tooltip.hh"

#include <cctype>

namespace Jetstream {

struct FlowgraphConfigVectorInlineField {
    using Config = FlowgraphConfigFieldConfig;

    Result update(Config config);
    Sakura::Component& component();

 private:
    void parseFormat();
    Result formatCurrentValue();
    static bool normalizeVectorInput(const std::string& text, std::string& normalized, std::string& error);
    bool applyBuffer(const std::string& nextBuffer);

    Config config;
    Parser::Map parsedFormat;
    std::any parsedValue;
    F32 multiplier = 1.0f;
    std::string valueType = "float";
    std::string unit;
    int precision = 2;
    std::string buffer = "[]";
    FlowgraphFieldTooltip help;
    Sakura::NodeField frame;
    Sakura::NodeTextInput input;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_INLINE_HH
