#include "module_impl.hh"

#include <cmath>

namespace Jetstream::Modules {

namespace {

constexpr F32 kMinIntervalMs = 0.001f;
constexpr F32 kMaxIntervalMs = 86'400'000.0f;

}  // namespace

Result ThrottleImpl::validate() {
    const auto& config = *candidate();

    if (!std::isfinite(config.intervalMs) || config.intervalMs < kMinIntervalMs ||
        config.intervalMs > kMaxIntervalMs) {
        JST_ERROR("[MODULE_THROTTLE] Interval must be between 0.001 ms and one day.");
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
    outputs()["buffer"].produced(name(), "buffer", input);

    resetDeadline();

    return Result::SUCCESS;
}

Result ThrottleImpl::reconfigure() {
    const auto& config = *candidate();

    // Check if only interval changed.
    if (config.intervalMs != intervalMs) {
        intervalMs = config.intervalMs;
        resetDeadline();

        return Result::SUCCESS;
    }

    return Result::RECREATE;
}

void ThrottleImpl::resetDeadline() {
    const auto interval = std::chrono::ceil<std::chrono::steady_clock::duration>(
        std::chrono::duration<F64, std::milli>(intervalMs));
    schedule.reset(interval, std::chrono::steady_clock::now());
}

}  // namespace Jetstream::Modules
