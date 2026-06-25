#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_HH

#include "types.hh"

// TODO: Cleanup parsing.

namespace Jetstream {

struct FlowgraphConfigVectorField : public Sakura::Component {
    using Config = FlowgraphConfigFieldConfig;

    void update(Config config) {
        this->config = std::move(config);
        if (this->config.format != parsedFormat) {
            parseFormat();
        }
        if (this->config.encoded != parsedEncoded) {
            floatValues.clear();
            intValues.clear();
            if (!this->config.encoded.empty()) {
                if (valueType == "float") {
                    Parser::StringToTyped(this->config.encoded, floatValues);
                } else {
                    Parser::StringToTyped(this->config.encoded, intValues);
                }
            }
            parsedEncoded = this->config.encoded;
        }
        updateChildren();
    }

    void render(const Sakura::Context& ctx) const {
        if (valueType == "float") {
            for (U64 i = 0; i < floatInputs.size(); ++i) {
                floatFrames[i].render(ctx, [this, i](const Sakura::Context& ctx) {
                    floatInputs[i].render(ctx);
                });
            }
        } else if (valueType == "int") {
            for (U64 i = 0; i < intInputs.size(); ++i) {
                intFrames[i].render(ctx, [this, i](const Sakura::Context& ctx) {
                    intInputs[i].render(ctx);
                });
            }
        }
    }

 private:
    void parseFormat() {
        parsedFormat = config.format;
        const auto parts = Parser::SplitString(config.format, ":");
        valueType = (parts.size() > 1) ? parts[1] : "float";
        unit = (parts.size() > 2) ? parts[2] : "";
        precision = (parts.size() > 3 && !parts[3].empty()) ? std::stoi(parts[3]) : 2;
    }

    void updateChildren() {
        floatFrames.clear();
        floatInputs.clear();
        intFrames.clear();
        intInputs.clear();

        if (valueType == "float") {
            floatFrames.resize(floatValues.size());
            floatInputs.resize(floatValues.size());
            const F32 multiplier = ConfigUnitMultiplier(unit);
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
                        auto values = config.values;
                        auto nextValues = floatValues;
                        if (i < nextValues.size()) {
                            nextValues[i] = value * ConfigUnitMultiplier(unit);
                        }
                        values[config.name] = nextValues;
                        if (config.onApply) {
                            config.onApply(std::move(values), false);
                        }
                    },
                });
            }
        } else if (valueType == "int") {
            intFrames.resize(intValues.size());
            intInputs.resize(intValues.size());
            for (U64 i = 0; i < intValues.size(); ++i) {
                intFrames[i].update({
                    .id = config.id + "Frame" + std::to_string(i),
                    .label = config.label,
                    .help = config.help,
                });
                intInputs[i].update({
                    .id = config.id + "Int" + std::to_string(i),
                    .value = intValues[i],
                    .unit = unit,
                    .onChange = [this, i](U64 value) {
                        auto values = config.values;
                        auto nextValues = intValues;
                        if (i < nextValues.size()) {
                            nextValues[i] = value;
                        }
                        values[config.name] = nextValues;
                        if (config.onApply) {
                            config.onApply(std::move(values), false);
                        }
                    },
                });
            }
        }
    }

    Config config;
    std::string parsedFormat;
    std::string parsedEncoded;
    std::string valueType = "float";
    std::string unit;
    int precision = 2;
    std::vector<F32> floatValues;
    std::vector<U64> intValues;
    std::vector<Sakura::NodeField> floatFrames;
    std::vector<Sakura::NodeFloatInput> floatInputs;
    std::vector<Sakura::NodeField> intFrames;
    std::vector<Sakura::NodeIntInput> intInputs;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_VECTOR_HH
