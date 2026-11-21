#include "float.hh"

namespace Jetstream {

Result FlowgraphConfigFloatField::update(Config config) {
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
    help.update(this->config.id, this->config.help);
    frame.update({
        .id = this->config.id,
        .label = this->config.label,
        .tooltip = help.overlay(),
        .content = &input,
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

Sakura::Component& FlowgraphConfigFloatField::component() {
    return frame;
}

void FlowgraphConfigFloatField::parseFormat() {
    parsedFormat = config.format;
    unit = Parser::Get<std::string>(config.format, "unit");
    precision = Parser::Get<I32>(config.format, "precision", 2);
    multiplier = Parser::Get<F64>(config.format, "scale", 1.0);
    stepConfig = Parser::Get<std::string>(config.format, "step_config");
}

bool FlowgraphConfigFloatField::wide(const std::string& name) const {
    return config.values.contains(name) && config.values.at(name).type() == typeid(F64);
}

std::any FlowgraphConfigFloatField::patchValue(const std::string& name, const F64 next) const {
    if (wide(name)) {
        return next;
    }
    return static_cast<F32>(next);
}

}  // namespace Jetstream
