#include "module_impl.hh"

#include <cmath>
#include <exception>
#include <limits>

#include <jetstream/memory/axis.hh>
#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

Result FilterTapsImpl::validate() {
    const auto& config = *candidate();
    validatedOutputSizeBytes = 0;
    validatedSampleRateMetadata = 0.0f;
    validatedBandwidthMetadata = 0.0f;
    validatedCenterMetadata.clear();
    const auto narrowMetadata = [](const F64 value, F32& narrowed) {
        constexpr F64 maxF32 = static_cast<F64>(std::numeric_limits<F32>::max());
        if (value < -maxF32 || value > maxF32) {
            return false;
        }
        narrowed = static_cast<F32>(value);
        return std::isfinite(narrowed) && (value == 0.0 || narrowed != 0.0f);
    };

    if (!std::isfinite(config.sampleRate) || config.sampleRate <= 0.0) {
        JST_ERROR("[MODULE_FILTER_TAPS] Sample rate must be positive ({}).", config.sampleRate);
        return Result::ERROR;
    }
    F32 sampleRateMetadata = 0.0f;
    if (!narrowMetadata(config.sampleRate, sampleRateMetadata)) {
        JST_ERROR("[MODULE_FILTER_TAPS] Sample rate is not representable as nonzero F32 "
                  "metadata ({}).", config.sampleRate);
        return Result::ERROR;
    }

    if (!std::isfinite(config.bandwidth) || config.bandwidth <= 0.0) {
        JST_ERROR("[MODULE_FILTER_TAPS] Bandwidth ({:.2f} MHz) must be between "
                  "0 and sample rate ({:.2f} MHz).",
                  config.bandwidth / 1e6, config.sampleRate / 1e6);
        return Result::ERROR;
    }
    F32 bandwidthMetadata = 0.0f;
    if (!narrowMetadata(config.bandwidth, bandwidthMetadata)) {
        JST_ERROR("[MODULE_FILTER_TAPS] Bandwidth is not representable as nonzero F32 "
                  "metadata ({}).", config.bandwidth);
        return Result::ERROR;
    }
    if (config.bandwidth > config.sampleRate) {
        JST_ERROR("[MODULE_FILTER_TAPS] Bandwidth ({:.2f} MHz) must be between "
                  "0 and sample rate ({:.2f} MHz).",
                  config.bandwidth / 1e6, config.sampleRate / 1e6);
        return Result::ERROR;
    }

    if (config.taps == 0) {
        JST_ERROR("[MODULE_FILTER_TAPS] Number of taps cannot be zero.");
        return Result::ERROR;
    }

    if ((config.taps % 2) == 0) {
        JST_ERROR("[MODULE_FILTER_TAPS] Number of taps must be odd ({}).", config.taps);
        return Result::ERROR;
    }

    if (config.center.empty()) {
        JST_ERROR("[MODULE_FILTER_TAPS] At least one center frequency is required.");
        return Result::ERROR;
    }

    try {
        validatedCenterMetadata.resize(config.center.size());
    } catch (const std::exception&) {
        JST_ERROR("[MODULE_FILTER_TAPS] Failed to allocate center metadata.");
        return Result::ERROR;
    }

    const F64 halfSampleRate = config.sampleRate / 2.0;
    for (U64 i = 0; i < config.center.size(); ++i) {
        if (!std::isfinite(config.center[i])) {
            JST_ERROR("[MODULE_FILTER_TAPS] Center frequency #{} ({:.2f} MHz) must be "
                      "between {:.2f} MHz and {:.2f} MHz.",
                      i,
                      config.center[i] / 1e6,
                      -halfSampleRate / 1e6,
                      halfSampleRate / 1e6);
            return Result::ERROR;
        }
        F32 narrowedCenter = 0.0f;
        if (!narrowMetadata(config.center[i], narrowedCenter)) {
            JST_ERROR("[MODULE_FILTER_TAPS] Center frequency #{} is not representable "
                      "as F32 metadata ({}).", i, config.center[i]);
            return Result::ERROR;
        }
        if (config.center[i] > halfSampleRate ||
            config.center[i] < -halfSampleRate) {
            JST_ERROR("[MODULE_FILTER_TAPS] Center frequency #{} ({:.2f} MHz) must be "
                      "between {:.2f} MHz and {:.2f} MHz.",
                      i,
                      config.center[i] / 1e6,
                      -halfSampleRate / 1e6,
                      halfSampleRate / 1e6);
            return Result::ERROR;
        }
        validatedCenterMetadata[i] = narrowedCenter;
    }

    U64 outputElementCount = 0;
    U64 outputSizeBytes = 0;
    if (!detail::CheckedMultiply(static_cast<U64>(config.center.size()),
                                 config.taps,
                                 outputElementCount)) {
        JST_ERROR("[MODULE_FILTER_TAPS] Output shape exceeds the supported layout range.");
        return Result::ERROR;
    }
    if (!detail::CheckedMultiply(outputElementCount,
                                 static_cast<U64>(DataTypeSize(DataType::CF32)),
                                 outputSizeBytes)) {
        JST_ERROR("[MODULE_FILTER_TAPS] Output byte size exceeds the supported range.");
        return Result::ERROR;
    }

    validatedOutputSizeBytes = outputSizeBytes;
    validatedSampleRateMetadata = sampleRateMetadata;
    validatedBandwidthMetadata = bandwidthMetadata;
    return Result::SUCCESS;
}

Result FilterTapsImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::STATIC_OUTPUT));

    JST_CHECK(defineInterfaceOutput("coeffs"));

    return Result::SUCCESS;
}

Result FilterTapsImpl::create() {
    const U64 heads = center.size();

    JST_CHECK(coeffs.create(device(), DataType::CF32, {heads, taps}));
    JST_CHECK(SetSignalAxes(coeffs, {
        .sample = Index{1},
        .channel = Index{0},
    }));

    outputs()["coeffs"].produced(name(), "coeffs", coeffs);

    // Attach filter parameters as tensor attributes so downstream
    // blocks (e.g. FilterEngine) can read them.
    coeffs.setAttribute("sampleRate", validatedSampleRateMetadata);
    coeffs.setAttribute("bandwidth", validatedBandwidthMetadata);
    if (validatedCenterMetadata.size() == 1) {
        coeffs.setAttribute("center", validatedCenterMetadata.front());
    } else {
        coeffs.setAttribute("center", validatedCenterMetadata);
    }

    return Result::SUCCESS;
}

Result FilterTapsImpl::destroy() {
    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
