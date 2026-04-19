#include <cmath>

#include "module_impl.hh"

namespace Jetstream::Modules {

Result SquelchImpl::validate() {
    const auto& config = *candidate();

    if (config.threshold < 0.0f) {
        JST_ERROR("[MODULE_SQUELCH] Invalid threshold '{}', must be non-negative.", config.threshold);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result SquelchImpl::define() {
    JST_CHECK(defineInterfaceInput("signal"));
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result SquelchImpl::create() {
    const Tensor& inputTensor = inputs().at("signal").tensor;

    input = inputTensor;
    output = input.clone();
    outputs()["signal"].produced(name(), "signal", output);

    passingState.publish(false);
    amplitudeState.publish(0.0f);

    return Result::SUCCESS;
}

Result SquelchImpl::destroy() {
    passingState.publish(false);
    amplitudeState.publish(0.0f);
    return Result::SUCCESS;
}

Result SquelchImpl::reconfigure() {
    const auto& config = *candidate();

    const F32 eps = 1e-6f;

    if (std::abs(config.threshold - threshold) > eps) {
        threshold = config.threshold;
    }

    return Result::SUCCESS;
}

bool SquelchImpl::getPassing() const {
    return passingState.get();
}

F32 SquelchImpl::getAmplitude() const {
    return amplitudeState.get();
}

}  // namespace Jetstream::Modules
