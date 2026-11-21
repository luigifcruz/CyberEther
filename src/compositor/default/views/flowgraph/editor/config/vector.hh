#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_HH

#include "types.hh"

#include "../field_tooltip.hh"

#include <atomic>
#include <memory>
#include <type_traits>

namespace Jetstream {

struct FlowgraphConfigVectorField {
    using Config = FlowgraphConfigFieldConfig;

    Result update(Config config);
    Sakura::Component& component();

 private:
    struct Pending {
        std::vector<F32> floatValues;
        std::vector<U64> uintValues;
        std::vector<std::string> stringValues;
        std::atomic<bool> complete{false};
    };

    template<typename T>
    void submit(Parser::Map patch, std::vector<T> values);
    void parseFormat();
    std::string entryLabel(U64 index) const;
    void updateChildren();

    Config config;
    std::shared_ptr<Pending> pending;
    Parser::Map parsedFormat;
    std::any parsedValue;
    F64 multiplier = 1.0;
    std::string valueType = "float";
    std::string unit;
    std::string itemLabel;
    int precision = 2;
    std::vector<F32> floatValues;
    std::vector<U64> uintValues;
    std::vector<std::string> stringValues;
    std::vector<std::string> optionLabels;
    std::vector<std::string> optionValues;
    std::string builtType;
    FlowgraphFieldTooltip help;
    Sakura::NodeFieldGrid grid;
    std::vector<std::unique_ptr<Sakura::NodeField>> frames;
    std::vector<std::unique_ptr<Sakura::NodeFloatInput>> floatInputs;
    std::vector<std::unique_ptr<Sakura::NodeUIntInput>> uintInputs;
    std::vector<std::unique_ptr<Sakura::NodeCombo>> stringCombos;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_HH
