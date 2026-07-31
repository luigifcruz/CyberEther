#include "module_impl.hh"

namespace Jetstream::Modules {

Result AgcImpl::validate() {
    validatedSignalAxes = {};

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
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
