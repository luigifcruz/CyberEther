#include "tensor.hh"

namespace Jetstream {

Result FlowgraphConfigTensorField::update(Config config) {
    this->config = std::move(config);
    if (this->config.format != parsedFormat) {
        parseFormat();
    }
    JST_CHECK(readValues());
    help.update(this->config.id, this->config.help);
    frame.update({
        .id = this->config.id,
        .label = this->config.label,
        .tooltip = help.overlay(),
        .content = &row,
    });
    updateRow();
    return Result::SUCCESS;
}

Sakura::Component& FlowgraphConfigTensorField::component() {
    return frame;
}

const std::vector<FlowgraphConfigTensorField::Option>& FlowgraphConfigTensorField::DataTypeOptions() {
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

const std::vector<FlowgraphConfigTensorField::Option>& FlowgraphConfigTensorField::DeviceOptions() {
    static const std::vector<Option> options = {
        {"cpu", "CPU"},
        {"cuda", "CUDA"},
    };
    return options;
}

std::string FlowgraphConfigTensorField::labelForKey(const std::vector<Option>& options, const std::string& key) {
    for (const auto& option : options) {
        if (option.key == key) {
            return option.label;
        }
    }
    return key;
}

std::string FlowgraphConfigTensorField::keyForLabel(const std::vector<Option>& options, const std::string& label) {
    for (const auto& option : options) {
        if (option.label == label) {
            return option.key;
        }
    }
    return label;
}

std::vector<std::string> FlowgraphConfigTensorField::labelsForOptions(const std::vector<Option>& options) {
    std::vector<std::string> labels;
    labels.reserve(options.size());
    for (const auto& option : options) {
        labels.push_back(option.label);
    }
    return labels;
}

void FlowgraphConfigTensorField::parseFormat() {
    parsedFormat = config.format;
    entryIndex = Parser::Get<U64>(config.format, "index", 0);
    source = Parser::Get<std::string>(config.format, "source");
}

Result FlowgraphConfigTensorField::readValues() {
    tensorSpecs.clear();
    if (config.values.contains(source)) {
        JST_CHECK(Parser::Deserialize(config.values, source, tensorSpecs));
    }

    const U64 requiredSize = entryIndex + 1;
    if (tensorSpecs.size() < requiredSize) {
        tensorSpecs.resize(requiredSize);
    }
    return Result::SUCCESS;
}

void FlowgraphConfigTensorField::apply(std::vector<TensorSpec> nextSpecs) const {
    Parser::Map patch;
    if (Parser::Serialize(patch, source, nextSpecs) != Result::SUCCESS) {
        return;
    }
    if (config.onApply) {
        config.onApply(std::move(patch), false, {});
    }
}

void FlowgraphConfigTensorField::applyShape(const U64 index, const std::string& value) const {
    auto nextSpecs = tensorSpecs;
    if (index < nextSpecs.size()) {
        nextSpecs[index].shape = value;
    }
    apply(std::move(nextSpecs));
}

void FlowgraphConfigTensorField::applyDataType(const U64 index, const std::string& label) const {
    auto nextSpecs = tensorSpecs;
    if (index < nextSpecs.size()) {
        nextSpecs[index].dtype = keyForLabel(DataTypeOptions(), label);
    }
    apply(std::move(nextSpecs));
}

void FlowgraphConfigTensorField::applyDevice(const U64 index, const std::string& label) const {
    auto nextSpecs = tensorSpecs;
    if (index < nextSpecs.size()) {
        nextSpecs[index].device = keyForLabel(DeviceOptions(), label);
    }
    apply(std::move(nextSpecs));
}

void FlowgraphConfigTensorField::applySignalAxes(const U64 index, const std::string& value) const {
    auto nextSpecs = tensorSpecs;
    if (index < nextSpecs.size()) {
        nextSpecs[index].axes = value;
    }
    apply(std::move(nextSpecs));
}

void FlowgraphConfigTensorField::updateRow() {
    const U64 index = entryIndex;
    const auto spec = index < tensorSpecs.size() ? tensorSpecs[index] : TensorSpec{};

    row.update({
        .id = config.id,
        .shape = spec.shape,
        .shapeHint = "shape",
        .shapeMinWidth = MinShapeWidth,
        .onShapeChange = [this, index](const std::string& value) {
            applyShape(index, value);
        },
        .axes = spec.axes,
        .axesHint = "[B, C, S]",
        .axesMinWidth = MinSignalAxesWidth,
        .onAxesChange = [this, index](const std::string& value) {
            applySignalAxes(index, value);
        },
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
    });
}

}  // namespace Jetstream
