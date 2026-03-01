#include "module_impl.hh"

namespace Jetstream::Modules {

Result ThrottleImpl::validate() {
    const auto& config = *candidate();

    if (config.intervalMs == 0) {
        JST_ERROR("[MODULE_THROTTLE] Interval cannot be zero.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result ThrottleImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result ThrottleImpl::create() {
    const Tensor& inputTensor = inputs().at("buffer").tensor;
    input = inputTensor;

    // Install bypass - output is same as input.
    outputs()["buffer"] = {name(), "buffer", input};

    // Reset timing to allow immediate first pass.
    lastExecutionTime = std::chrono::steady_clock::now() -
                        std::chrono::milliseconds(intervalMs);

    return Result::SUCCESS;
}

Result ThrottleImpl::reconfigure() {
    const auto& config = *candidate();

    // Check if only interval changed.
    if (config.intervalMs != intervalMs) {
        intervalMs = config.intervalMs;

        // Reset timing to allow immediate pass after change.
        lastExecutionTime = std::chrono::steady_clock::now() -
                            std::chrono::milliseconds(intervalMs);

        return Result::SUCCESS;
    }

    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
