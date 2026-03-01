#include "module_impl.hh"

namespace Jetstream::Modules {

Result RangeImpl::validate() {
    const auto& config = *candidate();

    if (config.min >= config.max) {
        JST_ERROR("[MODULE_RANGE] Min ({}) must be less than max ({}).", config.min, config.max);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result RangeImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS));

    JST_CHECK(defineInterfaceOutput("signal"));

    JST_CHECK(defineInterfaceInput("signal"));

    return Result::SUCCESS;
}

Result RangeImpl::create() {
    const Tensor& inputTensor = inputs().at("signal").tensor;

    input = inputTensor;

    // Calculate scaling coefficients.

    updateCoefficients();

    // Allocate output tensor.

    JST_CHECK(output.create(input.device(), input.dtype(), input.shape()));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["signal"] = {name(), "signal", output};

    return Result::SUCCESS;
}

Result RangeImpl::destroy() {
    return Result::SUCCESS;
}

Result RangeImpl::reconfigure() {
    const auto& config = *candidate();

    min = config.min;
    max = config.max;

    updateCoefficients();

    return Result::SUCCESS;
}

void RangeImpl::updateCoefficients() {
    scalingCoeff = 1.0f / (max - min);
    offsetCoeff = -min * scalingCoeff;
}

}  // namespace Jetstream::Modules
