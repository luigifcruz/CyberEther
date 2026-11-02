#include "module_impl.hh"

#include <algorithm>
#include <any>
#include <cmath>

#include "jetstream/memory/attributes.hh"

namespace Jetstream::Modules {

namespace {

constexpr F32 kMinIntervalMs = 0.001f;
constexpr F32 kMaxIntervalMs = 86'400'000.0f;

}  // namespace

Result ThrottleImpl::validate() {
    const auto& config = *candidate();

    if (config.pacing != "interval" && config.pacing != "sampleRate") {
        JST_ERROR("[MODULE_THROTTLE] Invalid pacing '{}'.", config.pacing);
        return Result::ERROR;
    }

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

    samplesPerBuffer = input.size();
    std::optional<U64> channel;
    if (ReadAttribute(input, std::string(ChannelAxisAttribute), channel) == Result::SUCCESS &&
        channel && *channel < input.rank() && input.shape(*channel) > 0) {
        samplesPerBuffer /= input.shape(*channel);
    }
    warnedSampleRate = false;

    resetDeadline();

    return Result::SUCCESS;
}

Result ThrottleImpl::reconfigure() {
    const auto& config = *candidate();

    pacing = config.pacing;
    intervalMs = config.intervalMs;
    warnedSampleRate = false;
    resetDeadline();

    return Result::SUCCESS;
}

void ThrottleImpl::resetDeadline() {
    const auto interval = std::chrono::ceil<std::chrono::steady_clock::duration>(
        std::chrono::duration<F64, std::milli>(intervalMs));
    schedule.reset(pacing == "interval" ? interval : std::chrono::steady_clock::duration{},
                   std::chrono::steady_clock::now());
}

std::chrono::steady_clock::duration ThrottleImpl::bufferDuration() {
    std::optional<F64> sampleRate;
    const std::string sampleRateKey(SampleRateAttribute);
    if (input.hasAttribute(sampleRateKey)) {
        const auto value = input.attribute(sampleRateKey);
        if (const auto* number = std::any_cast<F32>(&value)) {
            sampleRate = *number;
        } else if (const auto* number = std::any_cast<F64>(&value)) {
            sampleRate = *number;
        }
    }

    if (!sampleRate || !std::isfinite(*sampleRate) || *sampleRate <= 0.0) {
        if (!warnedSampleRate) {
            JST_WARN("[MODULE_THROTTLE] Input has no sample rate, so buffers pass unpaced.");
            warnedSampleRate = true;
        }
        return {};
    }

    const F64 seconds = std::min(static_cast<F64>(samplesPerBuffer) / *sampleRate,
                                 static_cast<F64>(kMaxIntervalMs) / 1e3);
    return std::chrono::ceil<std::chrono::steady_clock::duration>(std::chrono::duration<F64>(seconds));
}

}  // namespace Jetstream::Modules
