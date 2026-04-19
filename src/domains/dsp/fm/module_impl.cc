#include "module_impl.hh"

#include <cmath>

namespace Jetstream::Modules {

Result FmImpl::validate() {
    const auto& config = *candidate();

    if (config.sampleRate <= 0.0f) {
        JST_ERROR("[MODULE_FM] Sample rate must be positive.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result FmImpl::define() {
    JST_CHECK(defineInterfaceInput("signal"));
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result FmImpl::create() {
    const Tensor& inputTensor = inputs().at("signal").tensor;
    input = inputTensor;

    // Initialize coefficients.
    updateCoefficients();

    // Allocate output tensor (real F32).
    JST_CHECK(output.create(input.device(), DataType::F32, input.shape()));
    JST_CHECK(output.propagateAttributes(input));
    output.setAttribute("frequency", 0.0f);

    outputs()["signal"].produced(name(), "signal", output);

    return Result::SUCCESS;
}

void FmImpl::updateCoefficients() {
    kf = 100e3f / sampleRate;
    ref = 1.0f / (2.0f * JST_PI * kf);
}

}  // namespace Jetstream::Modules
