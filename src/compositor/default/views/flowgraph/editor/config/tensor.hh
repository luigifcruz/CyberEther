#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TENSOR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TENSOR_HH

#include "types.hh"

#include "../field_tooltip.hh"

namespace Jetstream {

struct FlowgraphConfigTensorField {
    using Config = FlowgraphConfigFieldConfig;

    Result update(Config config);
    Sakura::Component& component();

 private:
    static constexpr F32 DataTypeWidth = 84.0f;
    static constexpr F32 DeviceWidth = 84.0f;
    static constexpr F32 MinShapeWidth = 56.0f;
    static constexpr F32 MinSignalAxesWidth = 88.0f;

    struct TensorSpec {
        std::string shape = "[1]";
        std::string axes = "";
        std::string dtype = "F32";
        std::string device = "cpu";

        JST_SERDES(shape, axes, dtype, device);
    };

    struct Option {
        std::string key;
        std::string label;
    };

    static const std::vector<Option>& DataTypeOptions();
    static const std::vector<Option>& DeviceOptions();
    static std::string labelForKey(const std::vector<Option>& options, const std::string& key);
    static std::string keyForLabel(const std::vector<Option>& options, const std::string& label);
    static std::vector<std::string> labelsForOptions(const std::vector<Option>& options);
    void parseFormat();
    Result readValues();
    void apply(std::vector<TensorSpec> nextSpecs) const;
    void applyShape(const U64 index, const std::string& value) const;
    void applyDataType(const U64 index, const std::string& label) const;
    void applyDevice(const U64 index, const std::string& label) const;
    void applySignalAxes(const U64 index, const std::string& value) const;
    void updateRow();

    Config config;
    Parser::Map parsedFormat;
    std::string source;
    U64 entryIndex = 0;
    std::vector<TensorSpec> tensorSpecs;
    Sakura::NodeInputRow row;
    FlowgraphFieldTooltip help;
    Sakura::NodeField frame;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TENSOR_HH
