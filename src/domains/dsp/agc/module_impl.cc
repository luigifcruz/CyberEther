#include "module_impl.hh"

#include <cmath>

namespace Jetstream::Modules {

Result AgcImpl::validate() {
    validatedSignalAxes = {};

    const auto& config = *candidate();
    if (config.tileSize == 0) {
        JST_ERROR("[MODULE_AGC] Tile size must be greater than zero.");
        return Result::ERROR;
    }
    if (!std::isfinite(config.reference) || config.reference <= 0.0) {
        JST_ERROR("[MODULE_AGC] Reference must be finite and positive.");
        return Result::ERROR;
    }
    if (!std::isfinite(config.epsilon) || config.epsilon <= 0.0) {
        JST_ERROR("[MODULE_AGC] Epsilon must be finite and positive.");
        return Result::ERROR;
    }
    if (!std::isfinite(config.minGain) || config.minGain <= 0.0) {
        JST_ERROR("[MODULE_AGC] Minimum gain must be finite and positive.");
        return Result::ERROR;
    }
    if (!std::isfinite(config.maxGain) || config.maxGain < config.minGain) {
        JST_ERROR("[MODULE_AGC] Maximum gain must be finite and no less than "
                  "minimum gain.");
        return Result::ERROR;
    }
    if (!std::isfinite(config.maxGainChange) || config.maxGainChange < 1.0) {
        JST_ERROR("[MODULE_AGC] Maximum gain change must be finite and at "
                  "least one.");
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
        JST_ERROR("[MODULE_AGC] Input must contain valid signal axis metadata.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result AgcImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::STATELESS));

    JST_CHECK(defineInterfaceInput("signal"));
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result AgcImpl::create() {
    const Tensor& inputTensor = inputs().at("signal").tensor;

    input = inputTensor;
    sampleAxis = *validatedSignalAxes.sample;
    laneCount = input.size() / input.shape(sampleAxis);

    // Allocate output tensor with same shape as input.
    JST_CHECK(output.create(input.device(), input.dtype(), input.shape()));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["signal"].produced(name(), "signal", output);

    return Result::SUCCESS;
}

Result AgcImpl::destroy() {
    return Result::SUCCESS;
}

Result AgcImpl::reconfigure() {
    const auto& config = *candidate();
    tileSize = config.tileSize;
    reference = config.reference;
    epsilon = config.epsilon;
    minGain = config.minGain;
    maxGain = config.maxGain;
    maxGainChange = config.maxGainChange;

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
