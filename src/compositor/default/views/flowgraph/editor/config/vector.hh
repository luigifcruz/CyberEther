#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_HH

#include "types.hh"

#include "../field_tooltip.hh"

#include <memory>

namespace Jetstream {

struct FlowgraphConfigVectorField {
    using Config = FlowgraphConfigFieldConfig;

    Result update(Config config) {
        this->config = std::move(config);
        if (this->config.format != parsedFormat) {
            parseFormat();
        }
        const std::any nextValue = this->config.values.contains(this->config.name)
            ? this->config.values.at(this->config.name) : std::any{};
        if (!Parser::Equal(nextValue, parsedValue)) {
            parsedValue.reset();
            floatValues.clear();
            uintValues.clear();
            if (valueType == "float") {
                JST_CHECK(Parser::Deserialize(this->config.values, this->config.name, floatValues));
            } else if (valueType == "uint") {
                JST_CHECK(Parser::Deserialize(this->config.values, this->config.name, uintValues));
            }
            parsedValue = nextValue;
        }
        updateChildren();
        return Result::SUCCESS;
    }

    Sakura::Component& component() {
        return grid;
    }

 private:
    void parseFormat() {
        parsedFormat = config.format;
        valueType = Parser::Get<std::string>(config.format, "value_type", "float");
        unit = Parser::Get<std::string>(config.format, "unit");
        precision = Parser::Get<I32>(config.format, "precision", 2);
        multiplier = Parser::Get<F32>(config.format, "scale", 1.0f);
        parsedValue.reset();
        floatValues.clear();
        uintValues.clear();
    }

    void updateChildren() {
        const U64 count = valueType == "float" ? floatValues.size() : valueType == "uint" ? uintValues.size() : 0;
        const bool typeChanged = builtType != valueType;
        if (typeChanged || frames.size() != count) {
            frames.clear();
            floatInputs.clear();
            uintInputs.clear();
            for (U64 i = 0; i < count; ++i) {
                frames.push_back(std::make_unique<Sakura::NodeField>());
                if (valueType == "float") {
                    floatInputs.push_back(std::make_unique<Sakura::NodeFloatInput>());
                } else {
                    uintInputs.push_back(std::make_unique<Sakura::NodeUIntInput>());
                }
            }
            builtType = valueType;
        }

        help.update(config.id, config.help);
        std::vector<Sakura::NodeFieldGrid::Item> items;
        for (U64 i = 0; i < count; ++i) {
            frames[i]->update({
                .id = config.id + "Frame" + std::to_string(i),
                .label = config.label,
                .tooltip = help.overlay(),
                .content = valueType == "float" ? static_cast<Sakura::Component*>(floatInputs[i].get())
                                                : uintInputs[i].get(),
            });
            items.push_back({
                .child = frames[i].get(),
                .fullWidth = true,
            });
        }

        if (valueType == "float") {
            for (U64 i = 0; i < count; ++i) {
                floatInputs[i]->update({
                    .id = config.id + "Float" + std::to_string(i),
                    .value = floatValues[i] / multiplier,
                    .unit = unit,
                    .precision = precision,
                    .onChange = [this, i](F32 value) {
                        Parser::Map patch;
                        auto nextValues = floatValues;
                        if (i < nextValues.size()) {
                            nextValues[i] = value * multiplier;
                        }
                        patch[config.name] = nextValues;
                        if (config.onApply) {
                            config.onApply(std::move(patch), false);
                        }
                    },
                });
            }
        } else if (valueType == "uint") {
            for (U64 i = 0; i < count; ++i) {
                uintInputs[i]->update({
                    .id = config.id + "UInt" + std::to_string(i),
                    .value = uintValues[i],
                    .unit = unit,
                    .onChange = [this, i](U64 value) {
                        Parser::Map patch;
                        auto nextValues = uintValues;
                        if (i < nextValues.size()) {
                            nextValues[i] = value;
                        }
                        patch[config.name] = nextValues;
                        if (config.onApply) {
                            config.onApply(std::move(patch), false);
                        }
                    },
                });
            }
        }

        grid.update({
            .id = config.id + "Grid",
            .items = std::move(items),
        });
    }

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
