#include "module_impl.hh"

#include <jetstream/memory/axis.hh>

namespace Jetstream::Modules {

Result InvertImpl::validate() {
    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    SignalAxes axes;
    if (ResolveSignalAxes(inputTensor, axes) != Result::SUCCESS) {
        JST_ERROR("[MODULE_INVERT] Input must contain valid signal axis metadata.");
        return Result::ERROR;
    }

    resolvedAxis = *axes.sample;
    return Result::SUCCESS;
}

Result InvertImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS |
                          Module::Taint::STATELESS));

    JST_CHECK(defineInterfaceInput("signal"));
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result InvertImpl::create() {
    const Tensor& inputTensor = inputs().at("signal").tensor;

    input = inputTensor;

    JST_CHECK(output.create(input.device(), input.dtype(), input.shape()));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["signal"].produced(name(), "signal", output);

    return Result::SUCCESS;
}

Result InvertImpl::destroy() {
    return Result::SUCCESS;
}

Result InvertImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
