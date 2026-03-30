#include "module_impl.hh"

namespace Jetstream::Modules {

Result DuplicateImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS));

    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result DuplicateImpl::create() {
    const Tensor& inputTensor = inputs().at("buffer").tensor;

    input = inputTensor;

    // Allocate output tensor with same shape as input.
    JST_CHECK(output.create(input.device(), input.dtype(), input.shape()));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["buffer"].produced(name(), "buffer", output);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
