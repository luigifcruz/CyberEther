#include "module_impl.hh"

#include <cmath>
#include <cstring>

namespace Jetstream::Modules {

Result RrcFilterImpl::validate() {
    const auto& config = *candidate();

    if (config.symbolRate <= 0.0f) {
        JST_ERROR("[MODULE_RRC_FILTER] Symbol rate must be "
                  "positive ({}).", config.symbolRate);
        return Result::ERROR;
    }

    if (config.sampleRate <= 0.0f) {
        JST_ERROR("[MODULE_RRC_FILTER] Sample rate must be "
                  "positive ({}).", config.sampleRate);
        return Result::ERROR;
    }

    if (config.sampleRate <= config.symbolRate) {
        JST_ERROR("[MODULE_RRC_FILTER] Sample rate ({}) must be "
                  "greater than symbol rate ({}).",
                  config.sampleRate, config.symbolRate);
        return Result::ERROR;
    }

    if (config.rollOff < 0.0f || config.rollOff > 1.0f) {
        JST_ERROR("[MODULE_RRC_FILTER] Roll-off factor must be "
                  "between 0.0 and 1.0 ({}).", config.rollOff);
        return Result::ERROR;
    }

    if ((config.taps % 2) == 0) {
        JST_ERROR("[MODULE_RRC_FILTER] Number of taps must be "
                  "odd ({}).", config.taps);
        return Result::ERROR;
    }

    if (config.taps < 3) {
        JST_ERROR("[MODULE_RRC_FILTER] Number of taps must be "
                  "at least 3 ({}).", config.taps);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result RrcFilterImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result RrcFilterImpl::create() {
    const Tensor& inputTensor = inputs().at("buffer").tensor;

    input = inputTensor;

    // Allocate output tensor matching input shape and dtype.
    JST_CHECK(output.create(input.device(), input.dtype(), input.shape()));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["buffer"] = {name(), "buffer", output};

    // Allocate coefficient buffer (always F32).
    JST_CHECK(coeffs.create(input.device(), DataType::F32, {taps}));

    // Allocate history buffer matching input dtype.
    JST_CHECK(history.create(input.device(), input.dtype(), {taps}));

    // Zero the history buffer.
    std::memset(history.data(), 0, history.size() * history.elementSize());

    historyIndex = 0;

    // Generate initial RRC coefficients.
    JST_CHECK(generateCoefficients());

    return Result::SUCCESS;
}

Result RrcFilterImpl::reconfigure() {
    const auto& config = *candidate();

    // Taps change requires buffer reallocation.
    if (config.taps != taps) {
        return Result::RECREATE;
    }

    // Other parameter changes only need coefficient regeneration.
    symbolRate = config.symbolRate;
    sampleRate = config.sampleRate;
    rollOff = config.rollOff;

    JST_CHECK(generateCoefficients());

    return Result::SUCCESS;
}

Result RrcFilterImpl::generateCoefficients() {
    const F64 samplesPerSymbol = static_cast<F64>(sampleRate) / static_cast<F64>(symbolRate);
    const F64 beta = static_cast<F64>(rollOff);
    const F64 normFactor = std::sqrt(1.0 / samplesPerSymbol);

    F32* coeffPtr = coeffs.data<F32>();

    for (U64 i = 0; i < taps; ++i) {
        const F64 t = (static_cast<F64>(i) - static_cast<F64>(taps - 1) / 2.0) / samplesPerSymbol;

        F64 rrcValue;

        if (std::abs(t) < 1e-10) {
            // t = 0 case.
            rrcValue = normFactor * (1.0 + beta * (4.0 / JST_PI - 1.0));
        } else if (std::abs(std::abs(4.0 * beta * t) - 1.0) < 1e-10) {
            // Singularity case where denominator approaches zero.
            const F64 piOver4Beta = JST_PI / (4.0 * beta);
            rrcValue = normFactor * beta / std::sqrt(2.0) *
                       ((1.0 + 2.0 / JST_PI) * std::sin(piOver4Beta) +
                        (1.0 - 2.0 / JST_PI) * std::cos(piOver4Beta));
        } else {
            // General case.
            const F64 piT = JST_PI * t;
            const F64 fourBetaT = 4.0 * beta * t;
            const F64 denom = 1.0 - (fourBetaT * fourBetaT);

            rrcValue = normFactor *
                       (std::sin(piT * (1.0 - beta)) + 4.0 * beta * t * std::cos(piT * (1.0 + beta))) /
                       (piT * denom);
        }

        coeffPtr[i] = static_cast<F32>(rrcValue);
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
