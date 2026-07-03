#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TENSOR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TENSOR_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphConfigTensorField {
    using Config = FlowgraphConfigFieldConfig;

    void update(Config config) {
        this->config = std::move(config);
        if (this->config.format != parsedFormat) {
            parseFormat();
        }
        readValues();
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .help = this->config.help,
        });
        updateRow();
    }

    void render(const Sakura::Context& ctx) const {
        frame.render(ctx, [this](const Sakura::Context& ctx) {
            row.render(ctx);
        });
    }

 private:
    static constexpr const char* TensorSpecsKey = "outputTensorSpecs";
    static constexpr F32 DataTypeWidth = 84.0f;
    static constexpr F32 DeviceWidth = 84.0f;
    static constexpr F32 MinShapeWidth = 56.0f;

    struct TensorSpec {
        std::string shape = "[1]";
        std::string dtype = "F32";
        std::string device = "cpu";

        JST_SERDES(shape, dtype, device);
    };

    struct Option {
        std::string key;
        std::string label;
    };

    static const std::vector<Option>& DataTypeOptions() {
        static const std::vector<Option> options = {
            {"F32", "F32"},
            {"CF32", "CF32"},
            {"F64", "F64"},
            {"CF64", "CF64"},
            {"I8", "I8"},
            {"I16", "I16"},
            {"I32", "I32"},
            {"I64", "I64"},
            {"U8", "U8"},
            {"U16", "U16"},
            {"U32", "U32"},
            {"U64", "U64"},
        };
        return options;
    }

    static const std::vector<Option>& DeviceOptions() {
        static const std::vector<Option> options = {
            {"cpu", "CPU"},
            {"cuda", "CUDA"},
        };
        return options;
    }

    static std::string labelForKey(const std::vector<Option>& options, const std::string& key) {
        for (const auto& option : options) {
            if (option.key == key) {
                return option.label;
            }
        }
        return key;
    }

    static std::string keyForLabel(const std::vector<Option>& options, const std::string& label) {
        for (const auto& option : options) {
            if (option.label == label) {
                return option.key;
            }
        }
        return label;
    }

    static std::vector<std::string> labelsForOptions(const std::vector<Option>& options) {
        std::vector<std::string> labels;
        labels.reserve(options.size());
        for (const auto& option : options) {
            labels.push_back(option.label);
        }
        return labels;
    }

    void parseFormat() {
        parsedFormat = config.format;
        entryIndex = 0;

        const auto parts = Parser::SplitString(config.format, ":");
        if (parts.size() <= 1 || parts[1].empty()) {
            return;
        }

        try {
            entryIndex = static_cast<U64>(std::stoull(parts[1]));
        } catch (...) {
            entryIndex = 0;
        }
    }

    void readValues() {
        tensorSpecs.clear();
        if (config.values.contains(TensorSpecsKey)) {
            (void)Parser::Deserialize(config.values, TensorSpecsKey, tensorSpecs);
        }

        const U64 requiredSize = entryIndex + 1;
        if (tensorSpecs.size() < requiredSize) {
            tensorSpecs.resize(requiredSize);
        }
    }

    void apply(std::vector<TensorSpec> nextSpecs) const {
        auto values = config.values;
        if (Parser::Serialize(values, TensorSpecsKey, nextSpecs) != Result::SUCCESS) {
            return;
        }
        if (config.onApply) {
            config.onApply(std::move(values), false);
        }
    }

    void applyShape(const U64 index, const std::string& value) const {
        auto nextSpecs = tensorSpecs;
        if (index < nextSpecs.size()) {
            nextSpecs[index].shape = value;
        }
        apply(std::move(nextSpecs));
    }

    void applyDataType(const U64 index, const std::string& label) const {
        auto nextSpecs = tensorSpecs;
        if (index < nextSpecs.size()) {
            nextSpecs[index].dtype = keyForLabel(DataTypeOptions(), label);
        }
        apply(std::move(nextSpecs));
    }

    void applyDevice(const U64 index, const std::string& label) const {
        auto nextSpecs = tensorSpecs;
        if (index < nextSpecs.size()) {
            nextSpecs[index].device = keyForLabel(DeviceOptions(), label);
        }
        apply(std::move(nextSpecs));
    }

    void updateRow() {
        const U64 index = entryIndex;
        const auto spec = index < tensorSpecs.size() ? tensorSpecs[index] : TensorSpec{};

        row.update({
            .id = config.id,
            .value = spec.shape,
            .hint = "shape",
            .minInputWidth = MinShapeWidth,
            .combos = {
                {
                    .options = labelsForOptions(DataTypeOptions()),
                    .value = labelForKey(DataTypeOptions(), spec.dtype),
                    .width = DataTypeWidth,
                    .onChange = [this, index](const std::string& label) {
                        applyDataType(index, label);
                    },
                },
                {
                    .options = labelsForOptions(DeviceOptions()),
                    .value = labelForKey(DeviceOptions(), spec.device),
                    .width = DeviceWidth,
                    .onChange = [this, index](const std::string& label) {
                        applyDevice(index, label);
                    },
                },
            },
            .onChange = [this, index](const std::string& value) {
                applyShape(index, value);
            },
        });
    }

    Config config;
    std::string parsedFormat;
    U64 entryIndex = 0;
    std::vector<TensorSpec> tensorSpecs;
    Sakura::NodeInputRow row;
    Sakura::NodeField frame;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_TENSOR_HH
