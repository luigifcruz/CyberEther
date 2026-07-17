#include "module_impl.hh"

#include <cmath>
#include <limits>

namespace Jetstream::Modules {

Result AmplitudeImpl::validate() {
    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (inputTensor.rank() == 0 ||
        inputTensor.rank() > static_cast<U64>(std::numeric_limits<I64>::max())) {
        JST_ERROR("[MODULE_AMPLITUDE] Expected an input tensor with at least one dimension.");
        return Result::ERROR;
    }

    const auto& config = *candidate();
    const I64 rank = static_cast<I64>(inputTensor.rank());
    const I64 normalizedAxis = config.axis < 0 ? rank + config.axis : config.axis;
    if (normalizedAxis < 0 || normalizedAxis >= rank) {
        JST_ERROR("[MODULE_AMPLITUDE] Axis {} is out of bounds for a rank-{} tensor.",
                  config.axis,
                  rank);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result AmplitudeImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS));

    JST_CHECK(defineInterfaceInput("signal"));
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result AmplitudeImpl::create() {
    const Tensor& inputTensor = inputs().at("signal").tensor;

    input = inputTensor;

    if (input.rank() == 0 ||
        input.rank() > static_cast<U64>(std::numeric_limits<I64>::max())) {
        JST_ERROR("[MODULE_AMPLITUDE] Expected an input tensor with at least one dimension.");
        return Result::ERROR;
    }

    const I64 rank = static_cast<I64>(input.rank());
    const I64 normalizedAxis = axis < 0 ? rank + axis : axis;
    if (normalizedAxis < 0 || normalizedAxis >= rank) {
        JST_ERROR("[MODULE_AMPLITUDE] Axis {} is out of bounds for a rank-{} tensor.",
                  axis,
                  rank);
        return Result::ERROR;
    }

    const Index resolvedAxis = static_cast<Index>(normalizedAxis);
    scalingCoeff = 20.0f *
                   std::log10(1.0f / static_cast<F32>(input.shape(resolvedAxis)));

    // Create output tensor with same shape but F32 type.
    JST_CHECK(output.create(input.device(), DataType::F32, input.shape()));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["signal"].produced(name(), "signal", output);

    return Result::SUCCESS;
}

Result AmplitudeImpl::destroy() {
    return Result::SUCCESS;
}

Result AmplitudeImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
