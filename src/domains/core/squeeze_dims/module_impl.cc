#include "module_impl.hh"

#include <jetstream/memory/axis.hh>

namespace Jetstream::Modules {

Result SqueezeDimsImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result SqueezeDimsImpl::create() {
    const Tensor& inputTensor = inputs().at("buffer").tensor;

    const auto maybeResolvedAxis = ResolveAxis(axis, inputTensor.rank());
    if (!maybeResolvedAxis) {
        JST_ERROR("[MODULE_SQUEEZE_DIMS] Axis {} out of range for tensor with {} dimensions.",
                  axis, inputTensor.rank());
        return Result::ERROR;
    }
    const Index resolvedAxis = *maybeResolvedAxis;

    // Validate dimension at axis has size 1.
    if (inputTensor.shape(resolvedAxis) != 1) {
        JST_ERROR("[MODULE_SQUEEZE_DIMS] Cannot squeeze dimension {} (size {}). "
                  "Dimension must have size 1.",
                  axis, inputTensor.shape(resolvedAxis));
        return Result::ERROR;
    }

    input = inputTensor;
    output = input;

    JST_CHECK(output.squeezeDims(resolvedAxis));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["buffer"].produced(name(), "buffer", output);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
