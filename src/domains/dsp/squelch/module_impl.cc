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

    passingState.store(false, std::memory_order_relaxed);
    amplitudeState.store(0.0f, std::memory_order_relaxed);

    return Result::SUCCESS;
}

Result SquelchImpl::destroy() {
    passingState.store(false, std::memory_order_relaxed);
    amplitudeState.store(0.0f, std::memory_order_relaxed);
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

}  // namespace Jetstream::Modules
