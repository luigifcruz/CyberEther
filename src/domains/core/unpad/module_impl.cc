#include "module_impl.hh"

namespace Jetstream::Modules {

Result UnpadImpl::define() {
    JST_CHECK(defineInterfaceInput("padded"));
    JST_CHECK(defineInterfaceOutput("unpadded"));
    JST_CHECK(defineInterfaceOutput("pad"));

    return Result::SUCCESS;
}

Result UnpadImpl::create() {
    const Tensor& inputTensor = inputs().at("padded").tensor;

    // Validate axis is within valid range.
    if (axis >= inputTensor.rank()) {
        JST_ERROR("[MODULE_UNPAD] Axis {} out of range for tensor with {} dimensions.",
                  axis, inputTensor.rank());
        return Result::ERROR;
    }

    input = inputTensor;
    inputAxisSize = input.shape(axis);

    // Validate size doesn't exceed axis dimension.
    if (size > inputAxisSize) {
        JST_ERROR("[MODULE_UNPAD] Size {} exceeds axis dimension {}.",
                  size, inputAxisSize);
        return Result::ERROR;
    }

    unpadAxisSize = inputAxisSize - size;

    // Build output shapes.
    Shape unpadShape = input.shape();
    unpadShape[axis] = unpadAxisSize;

    Shape padShape = input.shape();
    padShape[axis] = size;

    JST_CHECK(outputUnpadded.create(input.device(), input.dtype(), unpadShape));
    JST_CHECK(outputPad.create(input.device(), input.dtype(), padShape));
    JST_CHECK(outputUnpadded.propagateAttributes(input));
    JST_CHECK(outputPad.propagateAttributes(input));

    outputs()["unpadded"].produced(name(), "unpadded", outputUnpadded);
    outputs()["pad"].produced(name(), "pad", outputPad);

    return Result::SUCCESS;

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
