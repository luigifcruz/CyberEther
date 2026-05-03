#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_FLOAT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_FLOAT_HH

#include "types.hh"

// TODO: Cleanup parsing.

namespace Jetstream {

struct FlowgraphConfigFloatField : public Sakura::Component {
    using Config = FlowgraphConfigFieldConfig;

    void update(Config config) {
        this->config = std::move(config);
        if (this->config.format != parsedFormat) {
            parseFormat();
        }
        if (this->config.encoded != parsedEncoded) {
            value = 0.0f;
            if (!this->config.encoded.empty()) {
                Parser::StringToTyped(this->config.encoded, value);
            }
            parsedEncoded = this->config.encoded;
        }

        std::string stepEncoded;
        if (!stepConfig.empty() && this->config.values.contains(stepConfig)) {
            Parser::TypedToString(this->config.values.at(stepConfig), stepEncoded);
        }
        if (stepEncoded != parsedStepEncoded) {
            step = 0.0f;
            hasStep = !stepConfig.empty() && stepConfig != this->config.name && this->config.values.contains(stepConfig) &&
                      Parser::Deserialize(this->config.values, stepConfig, step) == Result::SUCCESS;
            parsedStepEncoded = stepEncoded;
        }

        const F32 multiplier = ConfigUnitMultiplier(unit);
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .help = this->config.help,
        });
        input.update({
            .id = this->config.id + "Input",
            .value = value / multiplier,
            .unit = unit,
            .precision = precision,
            .step = hasStep ? std::optional<F32>(step / multiplier) : std::nullopt,
            .onChange = [this](F32 nextValue) {
                auto values = this->config.values;
                values[this->config.name] = nextValue * ConfigUnitMultiplier(unit);
                if (this->config.onApply) {
                    this->config.onApply(std::move(values), false);
                }
            },
            .onStepChange = [this](F32 nextStep) {
                if (stepConfig.empty()) {
                    return;
                }
                auto values = this->config.values;
                values[stepConfig] = nextStep * ConfigUnitMultiplier(unit);
                if (this->config.onApply) {
                    this->config.onApply(std::move(values), false);
                }
            },
        });
    }

    void render(const Sakura::Context& ctx) const {
        frame.render(ctx, [this](const Sakura::Context& ctx) {
            input.render(ctx);
        });
    }

 private:
    void parseFormat() {
        parsedFormat = config.format;
        const auto parts = Parser::SplitString(config.format, ":");
        unit = (parts.size() > 1) ? parts[1] : "";
        precision = (parts.size() > 2 && !parts[2].empty()) ? std::stoi(parts[2]) : 2;
        stepConfig = (parts.size() > 3) ? parts[3] : "";
        parsedStepEncoded.clear();
    }

    Config config;
    std::string parsedFormat;
    std::string parsedEncoded;
    std::string parsedStepEncoded;
    std::string unit;
    std::string stepConfig;
    int precision = 2;
    F32 value = 0.0f;
    F32 step = 0.0f;
    bool hasStep = false;
    Sakura::NodeField frame;
    Sakura::NodeFloatInput input;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_FLOAT_HH
