#include "jetstream/modules/rrc_filter.hh"

namespace Jetstream {

template<Device D, typename T>
struct RRCFilter<D, T>::Impl {
    using CoeffType = typename std::conditional_t<std::is_same_v<T, CF32>, F32, T>;
    Tensor<D, CoeffType> coeffs;
    Tensor<D, T> history;
    U64 historyIndex = 0;
    bool baked = false;

    Result generateRRCCoeffs(RRCFilter<D, T>& m);
    Result refreshCoefficients(RRCFilter<D, T>& m);
};

template<Device D, typename T>
Result RRCFilter<D, T>::Impl::generateRRCCoeffs(RRCFilter<D, T>& m) {
    const F64 samplesPerSymbol = m.config.sampleRate / m.config.symbolRate;
    const F64 beta = m.config.rollOff;
    const F64 normFactor = sqrt(1.0 / samplesPerSymbol);

    for (U64 i = 0; i < m.config.taps; i++) {
        const F64 t = (static_cast<F64>(i) - static_cast<F64>(m.config.taps - 1) / 2.0) / samplesPerSymbol;

        F64 rrcValue;

        if (abs(t) < 1e-10) {
            // t = 0 case
            rrcValue = normFactor * (1.0 + beta * (4.0 / JST_PI - 1.0));
        } else if (abs(abs(4.0 * beta * t) - 1.0) < 1e-10) {
            // Special case where denominator approaches zero
            const F64 piOver4Beta = JST_PI / (4.0 * beta);
            rrcValue = normFactor * beta / sqrt(2.0) *
                      ((1.0 + 2.0 / JST_PI) * sin(piOver4Beta) +
                       (1.0 - 2.0 / JST_PI) * cos(piOver4Beta));
        } else {
            // General case
            const F64 piT = JST_PI * t;
            const F64 fourBetaT = 4.0 * beta * t;
            const F64 denominator = 1.0 - (fourBetaT * fourBetaT);

            rrcValue = normFactor * (sin(piT * (1.0 - beta)) +
                                    4.0 * beta * t * cos(piT * (1.0 + beta))) /
                                   (piT * denominator);
        }

        coeffs[i] = static_cast<CoeffType>(rrcValue);
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result RRCFilter<D, T>::setSymbolRate(F32& symbolRate) {
    if (symbolRate <= 0) {
        JST_WARN("Invalid symbol rate: {} MHz. Symbol rate should be positive.",
                  symbolRate / JST_MHZ);
        symbolRate = config.symbolRate;
        return Result::WARNING;
    }
    config.symbolRate = symbolRate;
    return impl->refreshCoefficients(*this);
}

template<Device D, typename T>
Result RRCFilter<D, T>::setSampleRate(F32& sampleRate) {
    if (sampleRate <= 0) {
        JST_WARN("Invalid sample rate: {} MHz. Sample rate should be positive.",
                  sampleRate / JST_MHZ);
        sampleRate = config.sampleRate;
        return Result::WARNING;
    }
    if (sampleRate <= config.symbolRate) {
        JST_WARN("Sample rate ({} MHz) should be greater than symbol rate ({} MHz).",
                  sampleRate / JST_MHZ, config.symbolRate / JST_MHZ);
        sampleRate = config.sampleRate;
        return Result::WARNING;
    }
    config.sampleRate = sampleRate;
    return impl->refreshCoefficients(*this);
}

template<Device D, typename T>
Result RRCFilter<D, T>::setRollOff(F32& rollOff) {
    if (rollOff < 0.0f || rollOff > 1.0f) {
        JST_WARN("Invalid roll-off factor: {}. Roll-off should be between 0.0 and 1.0.",
                  rollOff);
        rollOff = config.rollOff;
        return Result::WARNING;
    }
    config.rollOff = rollOff;
    return impl->refreshCoefficients(*this);
}

template<Device D, typename T>
Result RRCFilter<D, T>::setTaps(U64& taps) {
    if ((taps % 2) == 0) {
        JST_WARN("Invalid number of taps: '{}'. Number of taps should be odd.", taps);
        taps = config.taps;
        return Result::WARNING;
    }
    if (taps < 3) {
        JST_WARN("Invalid number of taps: '{}'. Number of taps should be at least 3.", taps);
        taps = config.taps;
        return Result::WARNING;
    }

    // If taps changed, we need to reallocate buffers
    if (taps != config.taps) {
        config.taps = taps;

        // Reallocate coefficient buffer
        impl->coeffs = Tensor<D, typename Impl::CoeffType>({config.taps});

        // Reallocate and reset history buffer
        impl->history = Tensor<D, T>({config.taps});
        impl->historyIndex = 0;
        for (U64 i = 0; i < config.taps; i++) {
            impl->history[i] = T{};
        }

        return impl->refreshCoefficients(*this);
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result RRCFilter<D, T>::Impl::refreshCoefficients(RRCFilter<D, T>& m) {
    JST_CHECK(generateRRCCoeffs(m));
    baked = true;
    return Result::SUCCESS;
}

template<Device D, typename T>
Result RRCFilter<D, T>::create() {
    JST_DEBUG("Initializing RRC Filter module.");
    JST_INIT_IO();

    // Allocate filter coefficients
    impl->coeffs = Tensor<D, typename Impl::CoeffType>({config.taps});

    // Allocate history buffer (circular buffer for filter states)
    impl->history = Tensor<D, T>({config.taps});
    impl->historyIndex = 0;

    // Initialize history to zero
    for (U64 i = 0; i < config.taps; i++) {
        impl->history[i] = T{};
    }

    // Allocate output buffer to match input size
    output.buffer = Tensor<D, T>({input.buffer.size()});

    // Validate parameters
    if (config.symbolRate <= 0) {
        JST_ERROR("Invalid symbol rate: {} MHz. Symbol rate should be positive.",
                  config.symbolRate / JST_MHZ);
        return Result::ERROR;
    }
    if (config.sampleRate <= 0) {
        JST_ERROR("Invalid sample rate: {} MHz. Sample rate should be positive.",
                  config.sampleRate / JST_MHZ);
        return Result::ERROR;
    }
    if (config.sampleRate <= config.symbolRate) {
        JST_ERROR("Sample rate ({} MHz) should be greater than symbol rate ({} MHz).",
                  config.sampleRate / JST_MHZ, config.symbolRate / JST_MHZ);
        return Result::ERROR;
    }
    if (config.rollOff < 0.0f || config.rollOff > 1.0f) {
        JST_ERROR("Invalid roll-off factor: {}. Roll-off should be between 0.0 and 1.0.",
                  config.rollOff);
        return Result::ERROR;
    }
    if ((config.taps % 2) == 0) {
        JST_ERROR("Invalid number of taps: '{}'. Number of taps should be odd.", config.taps);
        return Result::ERROR;
    }
    if (config.taps < 3) {
        JST_ERROR("Invalid number of taps: '{}'. Number of taps should be at least 3.", config.taps);
        return Result::ERROR;
    }

    // Generate initial coefficients
    JST_CHECK(impl->generateRRCCoeffs(*this));
    impl->baked = true;

    return Result::SUCCESS;
}

template<Device D, typename T>
Result RRCFilter<D, T>::destroy() {
    return Result::SUCCESS;
}

template<Device D, typename T>
void RRCFilter<D, T>::info() const {
    JST_DEBUG("  Symbol Rate:   {} MHz", config.symbolRate / JST_MHZ);
    JST_DEBUG("  Sample Rate:   {} MHz", config.sampleRate / JST_MHZ);
    JST_DEBUG("  Roll-off:      {}", config.rollOff);
    JST_DEBUG("  Taps:          {}", config.taps);
    JST_DEBUG("  Oversampling:  {:.1f}", config.sampleRate / config.symbolRate);
}

}  // namespace Jetstream
