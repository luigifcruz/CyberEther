#include "module_impl.hh"

#include <cstring>

#include <jetstream/memory/axis.hh>
#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

Result OverlapAddImpl::validate() {
    validatedInputBuffer = Tensor();
    validatedInputOverlap = Tensor();
    validatedResolvedAxis = 0;
    validatedPreviousOverlapShape.clear();
    validatedOutputSizeBytes = 0;
    validatedPreviousOverlapSizeBytes = 0;

    if (!inputs().contains("buffer") || !inputs().contains("overlap")) {
        return Result::SUCCESS;
    }

    const Tensor& bufferTensor = inputs().at("buffer").tensor;
    const Tensor& overlapTensor = inputs().at("overlap").tensor;
    if (!bufferTensor.validShape() || !overlapTensor.validShape() ||
        bufferTensor.size() == 0 || overlapTensor.size() == 0) {
        return Result::SUCCESS;
    }

    const auto candidateAxis = ResolveAxis(candidate()->axis, bufferTensor.rank());
    if (!candidateAxis) {
        JST_ERROR("[MODULE_OVERLAP_ADD] Axis ({}) is out of "
                  "bounds for input rank ({}).",
                  candidate()->axis,
                  bufferTensor.rank());
        return Result::ERROR;
    }

    if (bufferTensor.rank() != overlapTensor.rank()) {
        JST_ERROR("[MODULE_OVERLAP_ADD] Buffer rank ({}) does "
                  "not match overlap rank ({}).",
                  bufferTensor.rank(),
                  overlapTensor.rank());
        return Result::ERROR;
    }

    if (bufferTensor.rank() > 1 && *candidateAxis == 0) {
        JST_ERROR("[MODULE_OVERLAP_ADD] Axis 0 is reserved for batch sequencing "
                  "for inputs with more than one dimension.");
        return Result::ERROR;
    }

    if (bufferTensor.shape(*candidateAxis) < overlapTensor.shape(*candidateAxis)) {
        JST_ERROR("[MODULE_OVERLAP_ADD] Overlap size ({}) is "
                  "larger than buffer size ({}) along axis ({}).",
                  overlapTensor.shape(*candidateAxis),
                  bufferTensor.shape(*candidateAxis),
                  *candidateAxis);
        return Result::ERROR;
    }

    for (Index dimension = 0; dimension < bufferTensor.rank(); ++dimension) {
        if (dimension == *candidateAxis) {
            continue;
        }

        if (bufferTensor.shape(dimension) != overlapTensor.shape(dimension)) {
            JST_ERROR("[MODULE_OVERLAP_ADD] Shape mismatch on axis "
                      "({}): buffer has {}, overlap has {}. Non-overlap "
                      "axes must match exactly.",
                      dimension,
                      bufferTensor.shape(dimension),
                      overlapTensor.shape(dimension));
            return Result::ERROR;
        }
    }

    Shape previousOverlapShape = overlapTensor.shape();
    U64 previousOverlapElementCount = overlapTensor.size();
    if (bufferTensor.rank() > 1) {
        previousOverlapElementCount /= previousOverlapShape[0];
        previousOverlapShape[0] = 1;
    }

    U64 outputSizeBytes = 0;
    U64 previousOverlapSizeBytes = 0;
    const U64 elementSize = static_cast<U64>(DataTypeSize(bufferTensor.dtype()));
    if (!detail::CheckedMultiply(bufferTensor.size(), elementSize, outputSizeBytes) ||
        !detail::CheckedMultiply(previousOverlapElementCount,
                                 elementSize,
                                 previousOverlapSizeBytes)) {
        JST_ERROR("[MODULE_OVERLAP_ADD] Output or previous-state byte size "
                  "exceeds the supported range.");
        return Result::ERROR;
    }

    validatedInputBuffer = bufferTensor;
    validatedInputOverlap = overlapTensor;
    validatedResolvedAxis = *candidateAxis;
    validatedPreviousOverlapShape = previousOverlapShape;
    validatedOutputSizeBytes = outputSizeBytes;
    validatedPreviousOverlapSizeBytes = previousOverlapSizeBytes;

    return Result::SUCCESS;
}

Result OverlapAddImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceInput("overlap"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result OverlapAddImpl::create() {
    inputBuffer = validatedInputBuffer;
    inputOverlap = validatedInputOverlap;

    // Allocate output tensor matching input buffer.
    JST_CHECK(output.create(inputBuffer.device(),
                            inputBuffer.dtype(),
                            inputBuffer.shape()));
    JST_CHECK(output.propagateAttributes(inputBuffer));

    // Allocate previous overlap state tensor.
    // Shape matches overlap but with batch dimension (dim 0) = 1.
    JST_CHECK(previousOverlap.create(inputBuffer.device(),
                                     inputBuffer.dtype(),
                                     validatedPreviousOverlapShape));

    // Zero the previous overlap.
    std::memset(previousOverlap.data(), 0, previousOverlap.sizeBytes());

    outputs()["buffer"].produced(name(), "buffer", output);

    return Result::SUCCESS;
}

Result OverlapAddImpl::destroy() {
    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
