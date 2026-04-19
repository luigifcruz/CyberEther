#include "module_impl.hh"

namespace Jetstream::Modules {

Result PadImpl::define() {
    JST_CHECK(defineInterfaceInput("unpadded"));
    JST_CHECK(defineInterfaceOutput("padded"));

    return Result::SUCCESS;
}

Result PadImpl::create() {
    const Tensor& inputTensor = inputs().at("unpadded").tensor;

    // Validate axis is within valid range.
    if (axis >= inputTensor.rank()) {
        JST_ERROR("[MODULE_PAD] Axis {} out of range for tensor with {} dimensions.",
                  axis, inputTensor.rank());
        return Result::ERROR;
    }

    input = inputTensor;
    inputAxisSize = input.shape(axis);
    outputAxisSize = inputAxisSize + size;

    // Build output shape with padding applied to the specified axis.
    Shape outputShape = input.shape();
    outputShape[axis] = outputAxisSize;

    JST_CHECK(output.create(input.device(), input.dtype(), outputShape));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["padded"].produced(name(), "padded", output);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
