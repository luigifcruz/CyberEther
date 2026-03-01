#include "module_impl.hh"

namespace Jetstream::Modules {

Result FilterTapsImpl::validate() {
    const auto& config = *candidate();

    if (config.sampleRate <= 0.0) {
        JST_ERROR("[MODULE_FILTER_TAPS] Sample rate must be positive ({}).", config.sampleRate);
        return Result::ERROR;
    }

    if (config.bandwidth <= 0.0 || config.bandwidth > config.sampleRate) {
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

    const F64 halfSampleRate = config.sampleRate / 2.0;
    for (U64 i = 0; i < config.center.size(); ++i) {
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
    }

    return Result::SUCCESS;
}

Result FilterTapsImpl::define() {
    JST_CHECK(defineInterfaceOutput("coeffs"));

    return Result::SUCCESS;
}

Result FilterTapsImpl::create() {
    const U64 heads = center.size();

    JST_CHECK(coeffs.create(device(), DataType::CF32, {heads, taps}));

    outputs()["coeffs"] = {name(), "coeffs", coeffs};

    // Attach filter parameters as tensor attributes so downstream
    // blocks (e.g. FilterEngine) can read them.
    coeffs.setAttribute("sampleRate", static_cast<F32>(sampleRate));
    coeffs.setAttribute("bandwidth", static_cast<F32>(bandwidth));
    coeffs.setAttribute("center", static_cast<F32>(center[0]));

    return Result::SUCCESS;
}

Result FilterTapsImpl::destroy() {
    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
