#include "module_impl.hh"

#include <algorithm>

namespace Jetstream::Modules {

Result RangeImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS | Module::Taint::STATELESS));

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

    outputs()["signal"].produced(name(), "signal", output);

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
    const F32 lower = std::min(min, max);
    const F32 upper = std::max(min, max);

    if (lower == upper) {
        scalingCoeff = 0.0f;
        offsetCoeff = 0.5f;
        return;
    }

    scalingCoeff = 1.0f / (upper - lower);
    offsetCoeff = -lower * scalingCoeff;
}

}  // namespace Jetstream::Modules
