#include "module_impl.hh"

#include <jetstream/memory/axis.hh>

namespace Jetstream::Modules {

Result UnpadImpl::validate() {
    validatedResolvedAxis = 0;
    validatedInputAxisSize = 0;
    validatedUnpadAxisSize = 0;

    if (!inputs().contains("padded")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("padded").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    const auto candidateAxis = ResolveAxis(candidate()->axis, inputTensor.rank());
    if (!candidateAxis) {
        JST_ERROR("[MODULE_UNPAD] Axis {} out of range for tensor with {} dimensions.",
                  candidate()->axis, inputTensor.rank());
        return Result::ERROR;
    }

    const U64 candidateInputAxisSize = inputTensor.shape(*candidateAxis);
    if (candidate()->size > candidateInputAxisSize) {
        JST_ERROR("[MODULE_UNPAD] Size {} exceeds axis dimension {}.",
                  candidate()->size, candidateInputAxisSize);
        return Result::ERROR;
    }

    validatedResolvedAxis = *candidateAxis;
    validatedInputAxisSize = candidateInputAxisSize;
    validatedUnpadAxisSize = candidateInputAxisSize - candidate()->size;
    return Result::SUCCESS;
}

Result UnpadImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::STATELESS));

    JST_CHECK(defineInterfaceInput("padded"));
    JST_CHECK(defineInterfaceOutput("unpadded"));
    JST_CHECK(defineInterfaceOutput("pad"));

    return Result::SUCCESS;
}

Result UnpadImpl::create() {
    const Tensor& inputTensor = inputs().at("padded").tensor;

    input = inputTensor;
    resolvedAxis = validatedResolvedAxis;
    inputAxisSize = validatedInputAxisSize;
    unpadAxisSize = validatedUnpadAxisSize;

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
