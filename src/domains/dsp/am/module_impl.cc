#include "module_impl.hh"

#include <cmath>

namespace Jetstream::Modules {

Result AmImpl::validate() {
    const auto& config = *candidate();

    if (config.sampleRate <= 0.0f) {
        JST_ERROR("[MODULE_AM] Sample rate must be positive.");
        return Result::ERROR;
    }

    if (config.dcAlpha < 0.0f || config.dcAlpha >= 1.0f) {
        JST_ERROR("[MODULE_AM] DC alpha must be in range [0, 1).");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result AmImpl::define() {
    JST_CHECK(defineInterfaceInput("signal"));
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result AmImpl::create() {
    const Tensor& inputTensor = inputs().at("signal").tensor;
    input = inputTensor;

    // Reset DC blocker state.
    prevEnvelope = 0.0f;
    prevOutput = 0.0f;

    // Allocate output tensor (real F32).
    JST_CHECK(output.create(input.device(), DataType::F32, input.shape()));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["signal"] = {name(), "signal", output};

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
