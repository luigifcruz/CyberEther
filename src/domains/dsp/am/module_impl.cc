#include "module_impl.hh"

#include <cmath>

namespace Jetstream::Modules {

Result AmImpl::validate() {
    validatedSignalAxes = {};
    validatedLaneCount = 0;

    const auto& config = *candidate();

    if (!std::isfinite(config.sampleRate) || config.sampleRate <= 0.0f) {
        JST_ERROR("[MODULE_AM] Sample rate must be finite and positive.");
        return Result::ERROR;
    }

    if (!std::isfinite(config.dcAlpha) ||
        config.dcAlpha < 0.0f || config.dcAlpha >= 1.0f) {
        JST_ERROR("[MODULE_AM] DC alpha must be in range [0, 1).");
        return Result::ERROR;
    }

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (ResolveSignalAxes(inputTensor, validatedSignalAxes) != Result::SUCCESS) {
        JST_ERROR("[MODULE_AM] Input must contain valid signal axis metadata.");
        return Result::ERROR;
    }

    validatedLaneCount = inputTensor.size() /
                         inputTensor.shape(*validatedSignalAxes.sample);
    if (validatedSignalAxes.batch) {
        validatedLaneCount /= inputTensor.shape(*validatedSignalAxes.batch);
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
    signalAxes = validatedSignalAxes;
    laneCount = validatedLaneCount;

    // Reset DC blocker state.
    prevEnvelope.assign(laneCount, 0.0f);
    prevOutput.assign(laneCount, 0.0f);

    // Allocate output tensor (real F32).
    JST_CHECK(output.create(input.device(), DataType::F32, input.shape()));
    JST_CHECK(output.propagateAttributes(input));
    JST_CHECK(SetSignalAxes(output, signalAxes));
    JST_CHECK(output.setAttribute("frequency", F32{0.0f}));

    outputs()["signal"].produced(name(), "signal", output);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
