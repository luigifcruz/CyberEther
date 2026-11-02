#include "module_impl.hh"

#include <cmath>
#include <optional>
#include <vector>

#include <jetstream/memory/attributes.hh>
#include <jetstream/memory/axis.hh>

namespace Jetstream::Modules {

Result PhaseCorrectionImpl::validate() {
    validatedBatchAxis.reset();
    validatedChannelAxis.reset();
    validatedChannelPhaseIncrements.clear();

    const auto& config = *candidate();
    if (!std::isfinite(config.phaseIncrement)) {
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

    std::optional<std::vector<F64>> typedIncrements;
    JST_CHECK(ReadAttribute(inputTensor, "channelPhaseIncrements", typedIncrements));
    const std::vector<F64> channelPhaseIncrements =
        typedIncrements.value_or(std::vector<F64>{});

    SignalAxes axes;
    if (ResolveSignalAxes(inputTensor, axes) != Result::SUCCESS) {
        JST_ERROR("[MODULE_PHASE_CORRECTION] Input signal axis metadata is invalid.");
        return Result::ERROR;
    }

    validatedBatchAxis = axes.batch;
    if (!channelPhaseIncrements.empty()) {
        if (!axes.channel ||
            channelPhaseIncrements.size() != inputTensor.shape(*axes.channel)) {
            JST_ERROR("[MODULE_PHASE_CORRECTION] Channel phase increments must "
                      "match channelAxis extent.");
            return Result::ERROR;
        }
        validatedChannelAxis = axes.channel;
        validatedChannelPhaseIncrements = channelPhaseIncrements;
    }
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
    channelAxis = validatedChannelAxis;
    channelPhaseIncrements = validatedChannelPhaseIncrements;

    JST_CHECK(output.create(input.device(), input.dtype(), input.shape()));
    JST_CHECK(output.propagateAttributes(input));
    JST_CHECK(output.removeAttribute("channelPhaseIncrements"));
    outputs()["signal"].produced(name(), "signal", output);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
