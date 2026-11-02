#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_FLOAT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_FLOAT_HH

#include "types.hh"

#include <cmath>
#include <limits>

namespace Jetstream {

struct FlowgraphConfigFloatField {
    using Config = FlowgraphConfigFieldConfig;

    Result update(Config config) {
        this->config = std::move(config);
        if (this->config.format != parsedFormat) {
            parseFormat();
        }
        value = 0.0f;
        JST_CHECK(Parser::Deserialize(this->config.values, this->config.name, value));
        step = Parser::Get<F32>(this->config.format, "step", 0.0f);
        hasStep = !stepConfig.empty() && stepConfig != this->config.name && this->config.values.contains(stepConfig) &&
                  Parser::Deserialize(this->config.values, stepConfig, step) == Result::SUCCESS &&
                  std::isfinite(step) && step > 0.0f;
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
            .step = (hasStep || step > 0.0f) ? std::optional<F32>(step / multiplier)
                                            : std::nullopt,
            .onChange = [this](F32 nextValue) {
                const F32 requested = nextValue * multiplier;
                if (!std::isfinite(requested) ||
                    requested < Parser::Get<F32>(this->config.format, "min",
                                                std::numeric_limits<F32>::lowest()) ||
                    requested > Parser::Get<F32>(this->config.format, "max",
                                                std::numeric_limits<F32>::max())) {
                    if (this->config.onError) {
                        this->config.onError(Result::ERROR,
                            this->config.label + ": Value is outside the allowed range.");
                    }
                    return;
                }
                Parser::Map patch;
                patch[this->config.name] = requested;
                if (this->config.onApply) {
                    this->config.onApply(std::move(patch), false, {});
                }
            },
            .onStepChange = [this](F32 nextStep) {
                if (stepConfig.empty()) {
                    return;
                }
                Parser::Map patch;
                patch[stepConfig] = nextStep * multiplier;
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
        multiplier = Parser::Get<F32>(config.format, "scale", 1.0f);
        stepConfig = Parser::Get<std::string>(config.format, "step_config");
    }

    Config config;
    Parser::Map parsedFormat;
    F32 multiplier = 1.0f;
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
