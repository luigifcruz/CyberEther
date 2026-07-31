#include "module_impl.hh"

#include <cmath>
#include <jetstream/memory/axis.hh>

namespace Jetstream::Modules {

Result AmplitudeImpl::validate() {
    validatedResolvedAxis = 0;

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    SignalAxes axes;
    if (ResolveSignalAxes(inputTensor, axes) != Result::SUCCESS) {
        JST_ERROR("[MODULE_AMPLITUDE] Input must contain valid signal axis metadata.");
        return Result::ERROR;
    }

    validatedResolvedAxis = *axes.sample;
    return Result::SUCCESS;
}

Result AmplitudeImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS | Module::Taint::STATELESS));

    JST_CHECK(defineInterfaceInput("signal"));
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result AmplitudeImpl::create() {
    const Tensor& inputTensor = inputs().at("signal").tensor;

    input = inputTensor;

    scalingCoeff = 20.0f *
                    std::log10(1.0f /
                               static_cast<F32>(input.shape(validatedResolvedAxis)));

    // Create output tensor with same shape but F32 type.
    JST_CHECK(output.create(input.device(), DataType::F32, input.shape()));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["signal"].produced(name(), "signal", output);

    return Result::SUCCESS;
}

Result AmplitudeImpl::destroy() {
    return Result::SUCCESS;
}

Result AmplitudeImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
