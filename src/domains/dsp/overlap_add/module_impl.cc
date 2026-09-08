#include "module_impl.hh"

#include <cstring>

#include <jetstream/memory/axis.hh>
#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

Result OverlapAddImpl::validate() {
    validatedInputBuffer = Tensor();
    validatedInputOverlap = Tensor();
    validatedBatchAxis.reset();
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

    if (bufferTensor.rank() != overlapTensor.rank()) {
        JST_ERROR("[MODULE_OVERLAP_ADD] Buffer rank ({}) does "
                  "not match overlap rank ({}).",
                  bufferTensor.rank(),
                  overlapTensor.rank());
        return Result::ERROR;
    }

    SignalAxes bufferAxes;
    SignalAxes overlapAxes;
    if (ResolveSignalAxes(bufferTensor, bufferAxes) != Result::SUCCESS ||
        ResolveSignalAxes(overlapTensor, overlapAxes) != Result::SUCCESS) {
        JST_ERROR("[MODULE_OVERLAP_ADD] Input signal axis metadata is invalid.");
        return Result::ERROR;
    }

    if (bufferAxes.sample != overlapAxes.sample ||
        bufferAxes.batch != overlapAxes.batch ||
        bufferAxes.channel != overlapAxes.channel) {
        JST_ERROR("[MODULE_OVERLAP_ADD] Buffer and overlap sample, batch, and "
                  "channel axes must match.");
        return Result::ERROR;
    }

    if (bufferTensor.shape(*bufferAxes.sample) <
        overlapTensor.shape(*overlapAxes.sample)) {
        JST_ERROR("[MODULE_OVERLAP_ADD] Overlap size ({}) is "
                  "larger than buffer size ({}) along axis ({}).",
                  overlapTensor.shape(*overlapAxes.sample),
                  bufferTensor.shape(*bufferAxes.sample),
                  *bufferAxes.sample);
        return Result::ERROR;
    }

    for (Index dimension = 0; dimension < bufferTensor.rank(); ++dimension) {
        if (dimension == *bufferAxes.sample) {
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
    if (bufferAxes.batch) {
        previousOverlapElementCount /= previousOverlapShape[*bufferAxes.batch];
        previousOverlapShape[*bufferAxes.batch] = 1;
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
    validatedBatchAxis = bufferAxes.batch;
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
    batchAxis = validatedBatchAxis;

    // Allocate output tensor matching input buffer.
    JST_CHECK(output.create(inputBuffer.device(),
                            inputBuffer.dtype(),
                            inputBuffer.shape()));
    JST_CHECK(output.propagateAttributes(inputBuffer));

    // Allocate previous overlap state tensor.
    // Shape matches overlap with the optional batch dimension reduced to one.
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
