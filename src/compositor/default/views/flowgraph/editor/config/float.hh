#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_FLOAT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_FLOAT_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphConfigFloatField {
    using Config = FlowgraphConfigFieldConfig;

    Result update(Config config) {
        this->config = std::move(config);
        if (this->config.format != parsedFormat) {
            parseFormat();
        }
        value = 0.0;
        JST_CHECK(Parser::Deserialize(this->config.values, this->config.name, value));
        step = 0.0;
        hasStep = !stepConfig.empty() && stepConfig != this->config.name && this->config.values.contains(stepConfig) &&
                  Parser::Deserialize(this->config.values, stepConfig, step) == Result::SUCCESS &&
                  std::isfinite(step) && step > 0.0;
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .help = this->config.help,
        });
        input.update({
            .id = this->config.id + "Input",
            .value = value,
            .scale = multiplier,
            .unit = unit,
            .precision = precision,
            .step = hasStep ? std::optional<F64>(step) : std::nullopt,
            .onChange = [this](F64 nextValue) {
                if (!wide(this->config.name) && static_cast<F32>(nextValue) == static_cast<F32>(value)) {
                    return;
                }
                Parser::Map patch;
                patch[this->config.name] = patchValue(this->config.name, nextValue);
                if (this->config.onApply) {
                    this->config.onApply(std::move(patch), false, {});
                }
            },
            .onStepChange = [this](F64 nextStep) {
                if (stepConfig.empty()) {
                    return;
                }
                Parser::Map patch;
                patch[stepConfig] = patchValue(stepConfig, nextStep);
                if (this->config.onApply) {
                    this->config.onApply(std::move(patch), false, {});
                }
            },
        });
        return Result::SUCCESS;
    }

    void render(const Sakura::Context& ctx) const {
        frame.render(ctx, [this](const Sakura::Context& ctx) {
            input.render(ctx);
        });
    }

 private:
    void parseFormat() {
        parsedFormat = config.format;
        unit = Parser::Get<std::string>(config.format, "unit");
        precision = Parser::Get<I32>(config.format, "precision", 2);
        multiplier = Parser::Get<F64>(config.format, "scale", 1.0);
        stepConfig = Parser::Get<std::string>(config.format, "step_config");
    }

    bool wide(const std::string& name) const {
        return config.values.contains(name) && config.values.at(name).type() == typeid(F64);
    }

    std::any patchValue(const std::string& name, const F64 next) const {
        if (wide(name)) {
            return next;
        }
        return static_cast<F32>(next);
    }

    Config config;
    Parser::Map parsedFormat;
    F64 multiplier = 1.0;
    std::string unit;
    std::string stepConfig;
    int precision = 2;
    F64 value = 0.0;
    F64 step = 0.0;
    bool hasStep = false;
    Sakura::NodeField frame;
    Sakura::NodeFloatInput input;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_FLOAT_HH
