#include "module_impl.hh"

#include <jetstream/memory/axis.hh>

namespace Jetstream::Modules {

Result FlattenImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result FlattenImpl::create() {
    const Tensor& inputTensor = inputs().at("buffer").tensor;
    SignalAxes inputAxes;
    JST_CHECK(MapSignalAxes(inputTensor, IdentityAxisMap(inputTensor.rank()), inputAxes));

    if (!inputTensor.contiguous()) {
        JST_ERROR("[MODULE_FLATTEN] Cannot flatten non-contiguous tensor. "
                  "Use the contiguous option or duplicate the tensor first.");
        return Result::ERROR;
    }

    input = inputTensor;
    output = input.clone();

    JST_CHECK(output.reshape({inputTensor.size()}));
    JST_CHECK(output.propagateAttributes(input));

    if (input.shape() == output.shape()) {
        JST_CHECK(SetSignalAxes(output, inputAxes));
    } else {
        JST_CHECK(SetSignalAxes(output, {}));
    }

    outputs()["buffer"].produced(name(), "buffer", output);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
