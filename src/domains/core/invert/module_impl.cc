#include "module_impl.hh"

#include <limits>

#include <jetstream/memory/axis.hh>

namespace Jetstream::Modules {

Result InvertImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS |
                          Module::Taint::STATELESS));

    JST_CHECK(defineInterfaceInput("signal"));
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result InvertImpl::create() {
    const Tensor& inputTensor = inputs().at("signal").tensor;

    input = inputTensor;

    const auto maybeResolvedAxis = ResolveAxis(axis, input.rank());
    if (!maybeResolvedAxis) {
        if (input.rank() == 0 ||
            input.rank() > static_cast<U64>(std::numeric_limits<I64>::max())) {
            JST_ERROR("[MODULE_INVERT] Expected an input tensor with at least one dimension.");
            return Result::ERROR;
        }

        const I64 rank = static_cast<I64>(input.rank());
        JST_ERROR("[MODULE_INVERT] Axis {} is out of bounds for a rank-{} tensor.", axis, rank);
        return Result::ERROR;
    }
    resolvedAxis = *maybeResolvedAxis;

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
