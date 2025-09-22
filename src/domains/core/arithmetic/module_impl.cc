#include "module_impl.hh"

namespace Jetstream::Modules {

Result ArithmeticImpl::validate() {
    const auto& config = *candidate();

    if (config.operation != "add" &&
        config.operation != "sub" &&
        config.operation != "mul" &&
        config.operation != "div") {
        JST_ERROR("[MODULE_ARITHMETIC] Invalid operation '{}'.", config.operation);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result ArithmeticImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS));

    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result ArithmeticImpl::create() {
    const Tensor& inputTensor = inputs().at("buffer").tensor;
    input = inputTensor;

    // Check input rank.

    if (input.rank() == 0) {
        JST_ERROR("[MODULE_ARITHMETIC] Input buffer rank is 0.");
        return Result::ERROR;
    }

    if (input.rank() <= axis) {
        JST_ERROR("[MODULE_ARITHMETIC] Input buffer rank {} is less than or "
                  "equal to axis {}.", input.rank(), axis);
        return Result::ERROR;
    }

    if (input.shape(axis) == 0) {
        JST_ERROR("[MODULE_ARITHMETIC] Input buffer axis {} is 0.", axis);
        return Result::ERROR;
    }

    // Calculate output shape.

    Shape outputShape(input.shape());
    outputShape[axis] = 1;

    const DeviceType device = input.device();
    const DataType dtype = input.dtype();

    // Allocate output.

    JST_CHECK(output.create(device, dtype, outputShape));

    // Create broadcast view for accumulation.

    broadcastedOutput = output.clone();
    JST_CHECK(broadcastedOutput.broadcastTo(input.shape()));

    // Apply squeeze if requested.

    if (squeeze) {
        JST_CHECK(output.squeezeDims(axis));
    }

    JST_CHECK(output.propagateAttributes(input));

    outputs()["buffer"] = {name(), "buffer", output};

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
