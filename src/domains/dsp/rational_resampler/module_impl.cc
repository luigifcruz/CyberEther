#include "module_impl.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

namespace {

std::any ResampledSampleRate(const Tensor& input, const U64 interpolation,
                            const U64 decimation) {
    if (!input.hasAttribute("sampleRate")) {
        return {};
    }
    const auto value = input.attribute("sampleRate");
    const auto* rate = std::any_cast<F32>(&value);
    if (!rate || !std::isfinite(*rate) || *rate <= 0.0f) {
        return {};
    }
    const F64 scaled = static_cast<F64>(*rate) *
                       (static_cast<F64>(interpolation) / decimation);
    if (!std::isfinite(scaled) || scaled > std::numeric_limits<F32>::max()) {
        return {};
    }
    const F32 result = static_cast<F32>(scaled);
    return result > 0.0f ? std::any(result) : std::any{};
}

}  // namespace

Result RationalResamplerImpl::validate() {
    validatedPlan = {};
    const auto& config = *candidate();
    if (config.interpolation == 0 || config.decimation == 0) {
        JST_ERROR("[MODULE_RATIONAL_RESAMPLER] Rate factors must be positive.");
        return Result::ERROR;
    }
    if (!std::isfinite(config.cutoff) || config.cutoff <= 0.0f ||
        config.cutoff >= 1.0f) {
        JST_ERROR("[MODULE_RATIONAL_RESAMPLER] Cutoff must be strictly between 0 and 1.");
        return Result::ERROR;
    }

    auto& p = validatedPlan;
    const U64 divisor = std::gcd(config.interpolation, config.decimation);
    p.interpolation = config.interpolation / divisor;
    p.decimation = config.decimation / divisor;
    const U64 rate = std::max(p.interpolation, p.decimation);
    U64 minimumTaps = 0;
    if (!detail::CheckedMultiply(rate, 2, minimumTaps) ||
        !detail::CheckedAdd(minimumTaps, 1, minimumTaps)) {
        JST_ERROR("[MODULE_RATIONAL_RESAMPLER] Rate factors exceed the filter size range.");
        return Result::ERROR;
    }
    p.tapCount = config.taps;
    if (p.tapCount == 0 &&
        (!detail::CheckedMultiply(rate, 64, p.tapCount) ||
         !detail::CheckedAdd(p.tapCount, 1, p.tapCount))) {
        JST_ERROR("[MODULE_RATIONAL_RESAMPLER] Automatic filter length is too large.");
        return Result::ERROR;
    }
    if (p.tapCount < minimumTaps || p.tapCount % 2 == 0) {
        JST_ERROR("[MODULE_RATIONAL_RESAMPLER] Taps must be odd and at least {} "
                  "(or zero for automatic).", minimumTaps);
        return Result::ERROR;
    }
    p.phaseLength = p.tapCount / p.interpolation +
                    (p.tapCount % p.interpolation != 0);

    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }
    const Tensor& tensor = inputs().at("buffer").tensor;
    if (!tensor.validShape() || tensor.size() == 0) {
        return Result::SUCCESS;
    }
    if (ResolveSignalAxes(tensor, p.axes) != Result::SUCCESS) {
        JST_ERROR("[MODULE_RATIONAL_RESAMPLER] Input must contain valid signal axes.");
        return Result::ERROR;
    }
    if (tensor.hasAttribute("sampleRate") &&
        !ResampledSampleRate(tensor, p.interpolation, p.decimation).has_value()) {
        JST_ERROR("[MODULE_RATIONAL_RESAMPLER] Sample rate must be positive F32 "
                  "with a representable resampled rate.");
        return Result::ERROR;
    }

    p.inputSamples = tensor.shape(*p.axes.sample);
    p.batchCount = p.axes.batch ? tensor.shape(*p.axes.batch) : 1;
    p.laneCount = tensor.size() / p.inputSamples / p.batchCount;
    U64 timeLimit = 0;
    if (!detail::CheckedMultiply(p.inputSamples, p.interpolation, p.duration) ||
        !detail::CheckedAdd(p.duration, p.decimation, timeLimit) ||
        !detail::CheckedAdd(p.inputSamples, p.phaseLength - 1, p.workspaceSize)) {
        JST_ERROR("[MODULE_RATIONAL_RESAMPLER] Input processing geometry is too large.");
        return Result::ERROR;
    }
    p.outputSamples = p.duration / p.decimation + (p.duration % p.decimation != 0);
    if (!detail::CheckedMultiply(p.outputSamples, p.batchCount, p.outputPerLane) ||
        !detail::CheckedMultiply(p.outputPerLane, 2, p.pendingCapacity)) {
        JST_ERROR("[MODULE_RATIONAL_RESAMPLER] Output processing geometry is too large.");
        return Result::ERROR;
    }
    p.outputShape = tensor.shape();
    p.outputShape[*p.axes.sample] = p.outputSamples;
    return Result::SUCCESS;
}

Result RationalResamplerImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS));
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));
    return Result::SUCCESS;
}

Result RationalResamplerImpl::create() {
    plan = validatedPlan;
    input = inputs().at("buffer").tensor;
    JST_CHECK(output.create(input.device(), input.dtype(), plan.outputShape));
    JST_CHECK(output.propagateAttributes(input));
    JST_CHECK(SetSignalAxes(output, plan.axes));
    JST_CHECK(updateSampleRate());
    outputs()["buffer"].produced(name(), "buffer", output);
    return Result::SUCCESS;
}

Result RationalResamplerImpl::updateSampleRate() {
    if (!input.hasAttribute("sampleRate")) {
        // Mask live inheritance until the next submission publishes a scaled rate.
        return output.removeAttribute("sampleRate");
    }
    const auto rate = ResampledSampleRate(input, plan.interpolation, plan.decimation);
    if (!rate.has_value()) {
        JST_ERROR("[MODULE_RATIONAL_RESAMPLER] Sample rate must be positive F32 "
                  "with a representable resampled rate.");
        return Result::ERROR;
    }
    return output.setAttribute("sampleRate", rate);
}

Result RationalResamplerImpl::reconfigure() {
    const auto& config = *candidate();
    if (config.interpolation != interpolation || config.decimation != decimation ||
        config.taps != taps || config.cutoff != cutoff) {
        // Queued samples and filter history belong to the previous configuration.
        return Result::RECREATE;
    }
    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
