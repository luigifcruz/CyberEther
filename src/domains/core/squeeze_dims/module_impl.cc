#include "module_impl.hh"

#include <jetstream/memory/axis.hh>

namespace Jetstream::Modules {

Result SqueezeDimsImpl::validate() {
    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("buffer").tensor;
    SignalAxes inputAxes;
    JST_CHECK(MapSignalAxes(inputTensor, IdentityAxisMap(inputTensor.rank()), inputAxes));

    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    const auto& config = *candidate();
    const auto candidateAxis = ResolveAxis(config.axis, inputTensor.rank());
    if (!candidateAxis) {
        JST_ERROR("[MODULE_SQUEEZE_DIMS] Axis {} out of range for tensor with {} dimensions.",
                  config.axis, inputTensor.rank());
        return Result::ERROR;
    }

    if (inputTensor.shape(*candidateAxis) != 1) {
        JST_ERROR("[MODULE_SQUEEZE_DIMS] Cannot squeeze dimension {} (size {}). "
                  "Dimension must have size 1.",
                  config.axis, inputTensor.shape(*candidateAxis));
        return Result::ERROR;
    }

    resolvedAxis = *candidateAxis;
    return Result::SUCCESS;
}

Result SqueezeDimsImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result SqueezeDimsImpl::create() {
    const Tensor& inputTensor = inputs().at("buffer").tensor;

    input = inputTensor;
    output = input.clone();

    JST_CHECK(output.squeezeDims(resolvedAxis));
    JST_CHECK(output.propagateAttributes(input));

    AxisMap axisMap(input.rank());
    for (Index inputAxis = 0; inputAxis < input.rank(); ++inputAxis) {
        if (inputAxis < resolvedAxis) {
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
