#include "module_impl.hh"

#include <limits>

namespace Jetstream::Modules {

Result InvertImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS));

    JST_CHECK(defineInterfaceInput("signal"));
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result InvertImpl::create() {
    const Tensor& inputTensor = inputs().at("signal").tensor;

    input = inputTensor;

    if (input.rank() == 0 ||
        input.rank() > static_cast<U64>(std::numeric_limits<I64>::max())) {
        JST_ERROR("[MODULE_INVERT] Expected an input tensor with at least one dimension.");
        return Result::ERROR;
    }

    const I64 rank = static_cast<I64>(input.rank());
    const I64 normalizedAxis = axis < 0 ? rank + axis : axis;
    if (normalizedAxis < 0 || normalizedAxis >= rank) {
        JST_ERROR("[MODULE_INVERT] Axis {} is out of bounds for a rank-{} tensor.", axis, rank);
        return Result::ERROR;
    }
    resolvedAxis = static_cast<Index>(normalizedAxis);

    JST_CHECK(output.create(input.device(), input.dtype(), input.shape()));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["signal"].produced(name(), "signal", output);

    return Result::SUCCESS;
}

Result InvertImpl::destroy() {
    return Result::SUCCESS;
}

Result InvertImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
