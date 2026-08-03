#include "module_impl.hh"

#include <cmath>

#include <jetstream/memory/axis.hh>

namespace Jetstream::Modules {

Result PhaseCorrectionImpl::validate() {
    validatedBatchAxis.reset();

    if (!std::isfinite(candidate()->phaseIncrement)) {
        JST_ERROR("[MODULE_PHASE_CORRECTION] Phase increment must be finite.");
        return Result::ERROR;
    }

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    SignalAxes axes;
    if (ResolveSignalAxes(inputTensor, axes) != Result::SUCCESS) {
        JST_ERROR("[MODULE_PHASE_CORRECTION] Input signal axis metadata is invalid.");
        return Result::ERROR;
    }

    validatedBatchAxis = axes.batch;
    return Result::SUCCESS;
}

Result PhaseCorrectionImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS));
    JST_CHECK(defineInterfaceInput("signal"));
    JST_CHECK(defineInterfaceOutput("signal"));
    return Result::SUCCESS;
}

Result PhaseCorrectionImpl::create() {
    input = inputs().at("signal").tensor;
    batchAxis = validatedBatchAxis;

    JST_CHECK(output.create(input.device(), input.dtype(), input.shape()));
    JST_CHECK(output.propagateAttributes(input));
    outputs()["signal"].produced(name(), "signal", output);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
