#include "module_impl.hh"

#include <jetstream/memory/axis.hh>

namespace Jetstream::Modules {

Result UnpadImpl::define() {
    JST_CHECK(defineInterfaceInput("padded"));
    JST_CHECK(defineInterfaceOutput("unpadded"));
    JST_CHECK(defineInterfaceOutput("pad"));

    return Result::SUCCESS;
}

Result UnpadImpl::create() {
    const Tensor& inputTensor = inputs().at("padded").tensor;

    const auto maybeResolvedAxis = ResolveAxis(axis, inputTensor.rank());
    if (!maybeResolvedAxis) {
        JST_ERROR("[MODULE_UNPAD] Axis {} out of range for tensor with {} dimensions.",
                  axis, inputTensor.rank());
        return Result::ERROR;
    }
    resolvedAxis = *maybeResolvedAxis;

    input = inputTensor;
    inputAxisSize = input.shape(resolvedAxis);

    // Validate size doesn't exceed axis dimension.
    if (size > inputAxisSize) {
        JST_ERROR("[MODULE_UNPAD] Size {} exceeds axis dimension {}.",
                  size, inputAxisSize);
        return Result::ERROR;
    }

    unpadAxisSize = inputAxisSize - size;

    // Build output shapes.
    Shape unpadShape = input.shape();
    unpadShape[resolvedAxis] = unpadAxisSize;

    Shape padShape = input.shape();
    padShape[resolvedAxis] = size;

    JST_CHECK(outputUnpadded.create(input.device(), input.dtype(), unpadShape));
    JST_CHECK(outputPad.create(input.device(), input.dtype(), padShape));
    JST_CHECK(outputUnpadded.propagateAttributes(input));
    JST_CHECK(outputPad.propagateAttributes(input));

    outputs()["unpadded"].produced(name(), "unpadded", outputUnpadded);
    outputs()["pad"].produced(name(), "pad", outputPad);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
