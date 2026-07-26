#include "module_impl.hh"

#include <limits>

#include <jetstream/memory/axis.hh>

namespace Jetstream::Modules {

Result InvertImpl::validate() {
    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    const auto& config = *candidate();
    const auto candidateAxis = ResolveAxis(config.axis, inputTensor.rank());
    if (!candidateAxis) {
        if (inputTensor.rank() == 0 ||
            inputTensor.rank() > static_cast<U64>(std::numeric_limits<I64>::max())) {
            JST_ERROR("[MODULE_INVERT] Expected an input tensor with at least one dimension.");
            return Result::ERROR;
        }

        JST_ERROR("[MODULE_INVERT] Axis {} is out of bounds for a rank-{} tensor.",
                  config.axis, inputTensor.rank());
        return Result::ERROR;
    }

    resolvedAxis = *candidateAxis;
    return Result::SUCCESS;
}

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
