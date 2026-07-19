#include "module_impl.hh"

#include <jetstream/memory/axis.hh>

namespace Jetstream::Modules {

Result ExpandDimsImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result ExpandDimsImpl::create() {
    const Tensor& inputTensor = inputs().at("buffer").tensor;

    const auto maybeResolvedAxis = ResolveInsertionAxis(axis, inputTensor.rank());
    if (!maybeResolvedAxis) {
        JST_ERROR("[MODULE_EXPAND_DIMS] Axis {} out of range for tensor with {} dimensions.",
                  axis, inputTensor.rank());
        return Result::ERROR;
    }
    const Index resolvedAxis = *maybeResolvedAxis;

    input = inputTensor;
    output = input;

    JST_CHECK(output.expandDims(resolvedAxis));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["buffer"].produced(name(), "buffer", output);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
