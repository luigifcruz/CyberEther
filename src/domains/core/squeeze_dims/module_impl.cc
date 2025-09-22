#include "module_impl.hh"

namespace Jetstream::Modules {

Result SqueezeDimsImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result SqueezeDimsImpl::create() {
    const Tensor& inputTensor = inputs().at("buffer").tensor;

    // Validate axis is within valid range [0, ndim-1].
    if (axis >= inputTensor.rank()) {
        JST_ERROR("[MODULE_SQUEEZE_DIMS] Axis {} out of range for tensor with {} dimensions.",
                  axis, inputTensor.rank());
        return Result::ERROR;
    }

    // Validate dimension at axis has size 1.
    if (inputTensor.shape(axis) != 1) {
        JST_ERROR("[MODULE_SQUEEZE_DIMS] Cannot squeeze dimension {} (size {}). "
                  "Dimension must have size 1.",
                  axis, inputTensor.shape(axis));
        return Result::ERROR;
    }

    input = inputTensor;
    output = input;

    JST_CHECK(output.squeezeDims(axis));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["buffer"] = {name(), "buffer", output};

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
