#include "module_impl.hh"

namespace Jetstream::Modules {

Result FftImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS));

    JST_CHECK(defineInterfaceInput("signal"));
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result FftImpl::create() {
    const Tensor& inputTensor = inputs().at("signal").tensor;

    input = inputTensor;

    // Create output tensor with same shape as input.
    // For real-to-complex FFT, output type will differ but shape remains same.
    JST_CHECK(output.create(input.device(), input.dtype(), input.shape()));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["signal"].produced(name(), "signal", output);

    return Result::SUCCESS;
}

Result FftImpl::destroy() {
    return Result::SUCCESS;
}

Result FftImpl::reconfigure() {
    // TODO: Implement update logic for FftImpl.
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
