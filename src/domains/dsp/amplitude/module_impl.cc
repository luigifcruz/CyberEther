#include "module_impl.hh"

#include <cmath>

namespace Jetstream::Modules {

Result AmplitudeImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS));

    JST_CHECK(defineInterfaceInput("signal"));
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result AmplitudeImpl::create() {
    const Tensor& inputTensor = inputs().at("signal").tensor;

    input = inputTensor;

    // Calculate scaling coefficient based on last axis size.
    const U64 lastAxis = input.rank() - 1;
    scalingCoeff = 20.0f * std::log10(1.0f / static_cast<F32>(input.shape()[lastAxis]));

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
