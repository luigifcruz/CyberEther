#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_HH

#include "types.hh"

#include "../field_tooltip.hh"

#include <memory>

namespace Jetstream {

struct FlowgraphConfigVectorField {
    using Config = FlowgraphConfigFieldConfig;

    Result update(Config config);
    Sakura::Component& component();

 private:
    void parseFormat();
    void updateChildren();

    Config config;
    Parser::Map parsedFormat;
    std::any parsedValue;
    F32 multiplier = 1.0f;
    std::string valueType = "float";
    std::string unit;
    int precision = 2;
    std::vector<F32> floatValues;
    std::vector<U64> uintValues;
    std::string builtType;
    FlowgraphFieldTooltip help;
    Sakura::NodeFieldGrid grid;
    std::vector<std::unique_ptr<Sakura::NodeField>> frames;
    std::vector<std::unique_ptr<Sakura::NodeFloatInput>> floatInputs;
    std::vector<std::unique_ptr<Sakura::NodeUIntInput>> uintInputs;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_HH
