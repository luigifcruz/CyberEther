#include "module_impl.hh"

#include <cmath>

namespace Jetstream::Modules {

Result FmImpl::validate() {
    validatedSignalAxes = {};
    validatedLaneCount = 0;

    const auto& config = *candidate();

    if (!std::isfinite(config.sampleRate) || config.sampleRate <= 0.0f) {
        JST_ERROR("[MODULE_FM] Sample rate must be finite and positive.");
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
        JST_ERROR("[MODULE_FM] Input must contain valid signal axis metadata.");
        return Result::ERROR;
    }

    validatedLaneCount = inputTensor.size() /
                         inputTensor.shape(*validatedSignalAxes.sample);
    if (validatedSignalAxes.batch) {
        validatedLaneCount /= inputTensor.shape(*validatedSignalAxes.batch);
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
    signalAxes = validatedSignalAxes;
    laneCount = validatedLaneCount;

    // Initialize coefficients.
    updateCoefficients();
    previousSample.assign(laneCount, CF32{0.0f, 0.0f});
    hasPreviousSample.assign(laneCount, U8{0});

    // Allocate output tensor (real F32).
    JST_CHECK(output.create(input.device(), DataType::F32, input.shape()));
    JST_CHECK(output.propagateAttributes(input));
    JST_CHECK(SetSignalAxes(output, signalAxes));
    JST_CHECK(output.setAttribute("frequency", F32{0.0f}));

    outputs()["signal"].produced(name(), "signal", output);

    return Result::SUCCESS;
}

void FmImpl::updateCoefficients() {
    kf = 100e3f / sampleRate;
    ref = 1.0f / (2.0f * JST_PI * kf);
}

}  // namespace Jetstream::Modules
