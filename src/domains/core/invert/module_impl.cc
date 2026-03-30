#include "module_impl.hh"

namespace Jetstream::Modules {

Result InvertImpl::define() {
    JST_CHECK(defineInterfaceInput("signal"));
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result InvertImpl::create() {
    const Tensor& inputTensor = inputs().at("signal").tensor;

    input = inputTensor;

    // Allocate output tensor with same shape as input.
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
