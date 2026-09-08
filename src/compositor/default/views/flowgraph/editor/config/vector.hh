#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_HH

#include "types.hh"

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

    void render(const Sakura::Context& ctx) const {
        if (valueType == "float") {
            for (U64 i = 0; i < floatInputs.size(); ++i) {
                floatFrames[i].render(ctx, [this, i](const Sakura::Context& ctx) {
                    floatInputs[i].render(ctx);
                });
            }
        } else if (valueType == "uint") {
            for (U64 i = 0; i < uintInputs.size(); ++i) {
                uintFrames[i].render(ctx, [this, i](const Sakura::Context& ctx) {
                    uintInputs[i].render(ctx);
                });
            }
        }
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
        floatFrames.clear();
        floatInputs.clear();
        uintFrames.clear();
        uintInputs.clear();

        if (valueType == "float") {
            floatFrames.resize(floatValues.size());
            floatInputs.resize(floatValues.size());
            for (U64 i = 0; i < floatValues.size(); ++i) {
                floatFrames[i].update({
                    .id = config.id + "Frame" + std::to_string(i),
                    .label = config.label,
                    .help = config.help,
                });
                floatInputs[i].update({
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
            uintFrames.resize(uintValues.size());
            uintInputs.resize(uintValues.size());
            for (U64 i = 0; i < uintValues.size(); ++i) {
                uintFrames[i].update({
                    .id = config.id + "Frame" + std::to_string(i),
                    .label = config.label,
                    .help = config.help,
                });
                uintInputs[i].update({
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
    std::vector<Sakura::NodeField> floatFrames;
    std::vector<Sakura::NodeFloatInput> floatInputs;
    std::vector<Sakura::NodeField> uintFrames;
    std::vector<Sakura::NodeUIntInput> uintInputs;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_HH
