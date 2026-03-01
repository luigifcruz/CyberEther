#include "module_impl.hh"

namespace Jetstream::Modules {

Result ExpandDimsImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result ExpandDimsImpl::create() {
    const Tensor& inputTensor = inputs().at("buffer").tensor;

    // Validate axis is within valid range [0, ndim].
    if (axis > inputTensor.rank()) {
        JST_ERROR("[MODULE_EXPAND_DIMS] Axis {} out of range for tensor with {} dimensions.",
                  axis, inputTensor.rank());
        return Result::ERROR;
    }

    input = inputTensor;
    output = input;

    JST_CHECK(output.expandDims(axis));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["buffer"] = {name(), "buffer", output};

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
