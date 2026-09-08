#include "module_impl.hh"

#include <jetstream/memory/axis.hh>

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

    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("buffer").tensor;
    SignalAxes inputAxes;
    JST_CHECK(MapSignalAxes(inputTensor, IdentityAxisMap(inputTensor.rank()), inputAxes));
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.rank() == 0) {
        JST_ERROR("[MODULE_ARITHMETIC] Input buffer rank is 0.");
        return Result::ERROR;
    }

    const auto candidateAxis = ResolveAxis(config.axis, inputTensor.rank());
    if (!candidateAxis) {
        JST_ERROR("[MODULE_ARITHMETIC] Axis {} out of range for input buffer rank {}.",
                  config.axis, inputTensor.rank());
        return Result::ERROR;
    }

    resolvedAxis = *candidateAxis;

    return Result::SUCCESS;
}

Result ArithmeticImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS | Module::Taint::STATELESS));

    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result ArithmeticImpl::create() {
    const Tensor& inputTensor = inputs().at("buffer").tensor;
    input = inputTensor;

    // Calculate output shape.

    Shape outputShape(input.shape());
    outputShape[resolvedAxis] = 1;

    const DeviceType device = input.device();
    const DataType dtype = input.dtype();

    // Allocate output.

    JST_CHECK(output.create(device, dtype, outputShape));

    // Create broadcast view for accumulation.

    broadcastedOutput = output.clone();
    JST_CHECK(broadcastedOutput.broadcastTo(input.shape()));

    // Apply squeeze if requested.

    if (squeeze) {
        JST_CHECK(output.squeezeDims(resolvedAxis));
    }

    JST_CHECK(output.propagateAttributes(input));

    AxisMap axisMap(input.rank());
    for (Index inputAxis = 0; inputAxis < input.rank(); ++inputAxis) {
        if (!squeeze || inputAxis < resolvedAxis) {
            axisMap[inputAxis] = inputAxis;
        } else if (inputAxis > resolvedAxis) {
            axisMap[inputAxis] = inputAxis - 1;
        }
    }
    SignalAxes outputAxes;
    JST_CHECK(MapSignalAxes(input, axisMap, outputAxes));
    JST_CHECK(SetSignalAxes(output, outputAxes));

    outputs()["buffer"].produced(name(), "buffer", output);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
