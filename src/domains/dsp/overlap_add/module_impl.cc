#include "module_impl.hh"

#include <cstring>

namespace Jetstream::Modules {

Result OverlapAddImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceInput("overlap"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result OverlapAddImpl::create() {
    const Tensor& bufferTensor = inputs().at("buffer").tensor;
    const Tensor& overlapTensor = inputs().at("overlap").tensor;

    inputBuffer = bufferTensor;
    inputOverlap = overlapTensor;

    // Validate axis bounds.
    if (axis >= inputBuffer.rank()) {
        JST_ERROR("[MODULE_OVERLAP_ADD] Axis ({}) is out of "
                  "bounds for input rank ({}).",
                  axis,
                  inputBuffer.rank());
        return Result::ERROR;
    }

    // Validate rank consistency.
    if (inputBuffer.rank() != inputOverlap.rank()) {
        JST_ERROR("[MODULE_OVERLAP_ADD] Buffer rank ({}) does "
                  "not match overlap rank ({}).",
                  inputBuffer.rank(),
                  inputOverlap.rank());
        return Result::ERROR;
    }

    // Validate overlap size.
    if (inputBuffer.shape(axis) < inputOverlap.shape(axis)) {
        JST_ERROR("[MODULE_OVERLAP_ADD] Overlap size ({}) is "
                  "larger than buffer size ({}) along axis ({}).",
                  inputOverlap.shape(axis),
                  inputBuffer.shape(axis),
                  axis);
        return Result::ERROR;
    }

    // Validate shape consistency on non-overlap axes.
    for (U64 d = 0; d < inputBuffer.rank(); ++d) {
        if (d == axis) {
            continue;
        }

        if (inputBuffer.shape(d) != inputOverlap.shape(d)) {
            JST_ERROR("[MODULE_OVERLAP_ADD] Shape mismatch on axis "
                      "({}): buffer has {}, overlap has {}. Non-overlap "
                      "axes must match exactly.",
                      d,
                      inputBuffer.shape(d),
                      inputOverlap.shape(d));
            return Result::ERROR;
        }
    }

    // Rank > 1 paths require at least one batch in overlap.
    if ((inputBuffer.rank() > 1) && (inputOverlap.shape(0) == 0)) {
        JST_ERROR("[MODULE_OVERLAP_ADD] Overlap batch dimension cannot "
                  "be zero for rank > 1 inputs.");
        return Result::ERROR;
    }

    // Allocate output tensor matching input buffer.
    JST_CHECK(output.create(inputBuffer.device(),
                            inputBuffer.dtype(),
                            inputBuffer.shape()));
    JST_CHECK(output.propagateAttributes(inputBuffer));

    outputs()["buffer"].produced(name(), "buffer", output);

    // Allocate previous overlap state tensor.
    // Shape matches overlap but with batch dimension (dim 0) = 1.
    auto prevShape = inputOverlap.shape();
    if (inputBuffer.rank() > 1) {
        prevShape[0] = 1;
    }
    JST_CHECK(previousOverlap.create(inputBuffer.device(),
                                     inputBuffer.dtype(),
                                     prevShape));

    // Zero the previous overlap.
    std::memset(previousOverlap.data(), 0, previousOverlap.size() *
                                           previousOverlap.elementSize());

    return Result::SUCCESS;
}

Result OverlapAddImpl::destroy() {
    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
