#include "module_impl.hh"

#include <jetstream/memory/axis.hh>

namespace Jetstream::Modules {

Result ExpandDimsImpl::validate() {
    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("buffer").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    const auto& config = *candidate();
    const auto candidateAxis = ResolveInsertionAxis(config.axis, inputTensor.rank());
    if (!candidateAxis) {
        JST_ERROR("[MODULE_EXPAND_DIMS] Axis {} out of range for tensor with {} dimensions.",
                  config.axis, inputTensor.rank());
        return Result::ERROR;
    }

    resolvedAxis = *candidateAxis;
    return Result::SUCCESS;
}

Result ExpandDimsImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result ExpandDimsImpl::create() {
    const Tensor& inputTensor = inputs().at("buffer").tensor;

    input = inputTensor;
    output = input;

    JST_CHECK(output.expandDims(resolvedAxis));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["buffer"].produced(name(), "buffer", output);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
