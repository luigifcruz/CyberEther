#include "module_impl.hh"

#include <jetstream/memory/axis.hh>

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
    JST_CHECK(defineTaint(Module::Taint::STATELESS));

    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result FoldImpl::create() {
    const Tensor& inputTensor = inputs().at("buffer").tensor;

    input = inputTensor;

    const auto candidateAxis = ResolveAxis(axis, input.rank());
    if (!candidateAxis) {
        JST_ERROR("[MODULE_FOLD] Axis ({}) is out of bounds for "
                  "input rank ({}).", axis, input.rank());
        return Result::ERROR;
    }
    resolvedAxis = *candidateAxis;

    // Validate size divides input dimension evenly.
    if (input.shape(resolvedAxis) % size != 0) {
        JST_ERROR("[MODULE_FOLD] Size ({}) is not a divisor of "
                  "the input shape ({}) along axis ({}).",
                  size, input.shape(resolvedAxis), resolvedAxis);
        return Result::ERROR;
    }

    // Validate offset bounds.
    if (input.shape(resolvedAxis) < offset) {
        JST_ERROR("[MODULE_FOLD] Offset ({}) is greater than the "
                  "input shape ({}) along axis ({}).",
                  offset, input.shape(resolvedAxis), resolvedAxis);
        return Result::ERROR;
    }

    // Calculate decimation factor.
    decimationFactor = input.shape(resolvedAxis) / size;

    // Build output shape.
    auto outputShape = input.shape();
    outputShape[resolvedAxis] = size;

    // Allocate output tensor with same dtype.
    JST_CHECK(output.create(input.device(), input.dtype(), outputShape));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["buffer"].produced(name(), "buffer", output);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
