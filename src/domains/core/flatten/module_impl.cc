#include "module_impl.hh"

namespace Jetstream::Modules {

Result FlattenImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result FlattenImpl::create() {
    const Tensor& inputTensor = inputs().at("buffer").tensor;

    if (!inputTensor.contiguous()) {
        JST_ERROR("[MODULE_FLATTEN] Cannot flatten non-contiguous tensor. "
                  "Use the contiguous option or duplicate the tensor first.");
        return Result::ERROR;
    }

    input = inputTensor;
    output = input;

    JST_CHECK(output.reshape({inputTensor.size()}));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["buffer"].produced(name(), "buffer", output);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
