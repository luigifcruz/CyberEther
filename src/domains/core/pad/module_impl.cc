#include "module_impl.hh"

#include <jetstream/memory/axis.hh>
#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

Result PadImpl::validate() {
    validatedResolvedAxis = 0;
    validatedInputAxisSize = 0;
    validatedOutputAxisSize = 0;
    validatedOutputSizeBytes = 0;

    if (!inputs().contains("unpadded")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("unpadded").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    const auto candidateAxis = ResolveAxis(candidate()->axis, inputTensor.rank());
    if (!candidateAxis) {
        JST_ERROR("[MODULE_PAD] Axis {} out of range for tensor with {} dimensions.",
                  candidate()->axis, inputTensor.rank());
        return Result::ERROR;
    }

    const U64 candidateInputAxisSize = inputTensor.shape(*candidateAxis);
    U64 candidateOutputAxisSize = 0;
    if (!detail::CheckedAdd(candidateInputAxisSize,
                            candidate()->size,
                            candidateOutputAxisSize)) {
        JST_ERROR("[MODULE_PAD] Padded axis size exceeds the supported range.");
        return Result::ERROR;
    }

    U64 outputElementCount = inputTensor.size() / candidateInputAxisSize;
    if (!detail::CheckedMultiply(outputElementCount,
                                 candidateOutputAxisSize,
                                 outputElementCount) ||
        !detail::CheckedMultiply(outputElementCount,
                                 static_cast<U64>(DataTypeSize(inputTensor.dtype())),
                                 validatedOutputSizeBytes)) {
        JST_ERROR("[MODULE_PAD] Padded output exceeds the supported layout range.");
        return Result::ERROR;
    }

    validatedResolvedAxis = *candidateAxis;
    validatedInputAxisSize = candidateInputAxisSize;
    validatedOutputAxisSize = candidateOutputAxisSize;
    return Result::SUCCESS;
}

Result PadImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::STATELESS));

    JST_CHECK(defineInterfaceInput("unpadded"));
    JST_CHECK(defineInterfaceOutput("padded"));

    return Result::SUCCESS;
}

Result PadImpl::create() {
    const Tensor& inputTensor = inputs().at("unpadded").tensor;

    input = inputTensor;
    resolvedAxis = validatedResolvedAxis;
    inputAxisSize = validatedInputAxisSize;
    outputAxisSize = validatedOutputAxisSize;

    // Build output shape with padding applied to the specified axis.
    Shape outputShape = input.shape();
    outputShape[resolvedAxis] = outputAxisSize;

    JST_CHECK(output.create(input.device(), input.dtype(), outputShape));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["padded"].produced(name(), "padded", output);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
