#include "module_impl.hh"

#include <algorithm>
#include <string>

namespace Jetstream::Modules {

Result SignalAxesImpl::validate() {
    validatedOverrideAxes = false;
    validatedAxes = {};

    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("buffer").tensor;
    if (!inputTensor.validShape()) {
        return Result::SUCCESS;
    }

    const auto& config = *candidate();
    SignalAxesLayout layout;
    if (ParseSignalAxesLayout(config.axes,
                              inputTensor.rank(),
                              SignalAxesLayoutMode::Transform,
                              layout) != Result::SUCCESS) {
        JST_ERROR("[MODULE_SIGNAL_AXES] Invalid axes layout.");
        return Result::ERROR;
    }

    if (!layout.specified) {
        Jetstream::SignalAxes inheritedAxes;
        if (MapSignalAxes(inputTensor,
                          IdentityAxisMap(inputTensor.rank()),
                          inheritedAxes) != Result::SUCCESS) {
            JST_ERROR("[MODULE_SIGNAL_AXES] Input contains invalid signal axis metadata.");
            return Result::ERROR;
        }
        return Result::SUCCESS;
    }

    validatedAxes = layout.axes;
    if (!layout.inherited.empty()) {
        Jetstream::SignalAxes inheritedAxes;
        if (MapSignalAxes(inputTensor,
                          IdentityAxisMap(inputTensor.rank()),
                          inheritedAxes) != Result::SUCCESS) {
            JST_ERROR("[MODULE_SIGNAL_AXES] Cannot inherit invalid input signal axis metadata.");
            return Result::ERROR;
        }
        if (!inputTensor.hasAttribute(std::string(SampleAxisAttribute))) {
            inheritedAxes.sample.reset();
        }

        const auto inheritRole = [&](const char role,
                                     const std::optional<Index>& inputAxis,
                                     std::optional<Index>& outputAxis) -> Result {
            if (!inputAxis ||
                std::find(layout.inherited.begin(),
                          layout.inherited.end(),
                          *inputAxis) == layout.inherited.end()) {
                return Result::SUCCESS;
            }
            if (outputAxis) {
                JST_ERROR("[MODULE_SIGNAL_AXES] Role '{}' is assigned to axis {} "
                          "and inherited from axis {}.",
                          role, *outputAxis, *inputAxis);
                return Result::ERROR;
            }
            outputAxis = inputAxis;
            return Result::SUCCESS;
        };

        JST_CHECK(inheritRole('B', inheritedAxes.batch, validatedAxes.batch));
        JST_CHECK(inheritRole('C', inheritedAxes.channel, validatedAxes.channel));
        JST_CHECK(inheritRole('S', inheritedAxes.sample, validatedAxes.sample));
    }
    validatedOverrideAxes = true;

    return Result::SUCCESS;
}

Result SignalAxesImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS |
                          Module::Taint::STATELESS));

    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result SignalAxesImpl::create() {
    const Tensor& inputTensor = inputs().at("buffer").tensor;

    input = inputTensor;
    output = input.clone();
    JST_CHECK(output.propagateAttributes(input));
    if (validatedOverrideAxes) {
        JST_CHECK(SetSignalAxes(output, validatedAxes));
    }

    outputs()["buffer"].produced(name(), "buffer", output);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
