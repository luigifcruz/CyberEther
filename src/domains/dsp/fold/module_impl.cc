#include "module_impl.hh"

namespace Jetstream::Modules {

Result FoldImpl::validate() {
    const auto& config = *candidate();

    if (config.size == 0) {
        JST_ERROR("[MODULE_FOLD] Size cannot be zero.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result FoldImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result FoldImpl::create() {
    const Tensor& inputTensor = inputs().at("buffer").tensor;

    input = inputTensor;

    // Validate axis bounds.
    if (input.rank() <= axis) {
        JST_ERROR("[MODULE_FOLD] Axis ({}) is out of bounds for "
                  "input rank ({}).", axis, input.rank());
        return Result::ERROR;
    }

    // Validate size divides input dimension evenly.
    if (input.shape(axis) % size != 0) {
        JST_ERROR("[MODULE_FOLD] Size ({}) is not a divisor of "
                  "the input shape ({}) along axis ({}).",
                  size, input.shape(axis), axis);
        return Result::ERROR;
    }

    // Validate offset bounds.
    if (input.shape(axis) < offset) {
        JST_ERROR("[MODULE_FOLD] Offset ({}) is greater than the "
                  "input shape ({}) along axis ({}).",
                  offset, input.shape(axis), axis);
        return Result::ERROR;
    }

    // Calculate decimation factor.
    decimationFactor = input.shape(axis) / size;

    // Build output shape.
    auto outputShape = input.shape();
    outputShape[axis] = size;

    // Allocate output tensor with same dtype.
    JST_CHECK(output.create(input.device(), input.dtype(), outputShape));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["buffer"].produced(name(), "buffer", output);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
