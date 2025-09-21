#include "jetstream/modules/psk_demod.hh"

#include <algorithm>
#include <deque>

namespace Jetstream {

template<Device D, typename T>
struct PskDemod<D, T>::Impl {
    // Configuration derived values
    U64 samplesPerSymbol;
    U64 constellationOrder;

    // PLL state for frequency/phase recovery
    F64 phaseAccumulator;
    F64 frequencyError;
    F64 freqAlpha;
    F64 freqBeta;

    // Timing recovery state
    F64 timingAlpha;
    F64 timingBeta;
    F64 timingMu;
    F64 timingOmega;
    F64 timingOmegaNominal;
    F64 timingOmegaMin;
    F64 timingOmegaMax;
    U64 timingIndex;

    // Symbol history for MM detector
    bool hasLastSymbol;
    T lastSymbol;
    T lastDecision;

    // Raw sample history for interpolation across buffers
    std::deque<T> sampleHistory;

    // Safety parameters
    static constexpr F64 MAX_TIMING_ERROR = 1.0;
    static constexpr F64 MIN_TIMING_ERROR = -1.0;
    static constexpr F64 MAX_FREQUENCY_ERROR = 1.0;
    static constexpr F64 MIN_FREQUENCY_ERROR = -1.0;

    // Helper methods
    T interpolate(const T& a, const T& b, F64 mu) const;
    T decision(const T& sample) const;
    F64 muellerMullerError(const T& prevSymbol, const T& prevDecision,
                           const T& currentSymbol, const T& currentDecision) const;
    F64 costasLoopError(const T& sample) const;
    T correctFrequency(const T& sample, F64 phase) const;
    void initializeParameters();
    Result refresh_values(const Config& config);
};

template<Device D, typename T>
Result PskDemod<D, T>::create() {
    JST_DEBUG("Initializing PSK Demod module.");
    JST_INIT_IO();

    // Check parameters.

    if (config.sampleRate <= 0) {
        JST_ERROR("Sample rate must be positive, got {}.", config.sampleRate);
        return Result::ERROR;
    }

    if (config.symbolRate <= 0) {
        JST_ERROR("Symbol rate must be positive, got {}.", config.symbolRate);
        return Result::ERROR;
    }

    if (config.symbolRate >= config.sampleRate) {
        JST_ERROR("Symbol rate {} must be less than sample rate {}.", config.symbolRate, config.sampleRate);
        return Result::ERROR;
    }

    if (config.frequencyLoopBandwidth <= 0 || config.frequencyLoopBandwidth >= 1.0) {
        JST_ERROR("Frequency loop bandwidth must be between 0 and 1, got {}.", config.frequencyLoopBandwidth);
        return Result::ERROR;
    }

    if (config.timingLoopBandwidth <= 0 || config.timingLoopBandwidth >= 1.0) {
        JST_ERROR("Timing loop bandwidth must be between 0 and 1, got {}.", config.timingLoopBandwidth);
        return Result::ERROR;
    }

    if (config.dampingFactor <= 0) {
        JST_ERROR("Damping factor must be positive, got {}.", config.dampingFactor);
        return Result::ERROR;
    }

    if (config.excessBandwidth < 0 || config.excessBandwidth > 1.0) {
        JST_ERROR("Excess bandwidth must be between 0 and 1, got {}.", config.excessBandwidth);
        return Result::ERROR;
    }

    if (input.buffer.rank() == 0) {
        JST_ERROR("Input buffer rank is 0.");
        return Result::ERROR;
    }

    if (config.bufferSize == 0) {
        JST_ERROR("Buffer size must be positive.");
        return Result::ERROR;
    }

    // Calculate samples per symbol
    pimpl->samplesPerSymbol = static_cast<U64>(config.sampleRate / config.symbolRate);
    if (pimpl->samplesPerSymbol < 2) {
        JST_ERROR("Samples per symbol must be at least 2, got {}.", pimpl->samplesPerSymbol);
        return Result::ERROR;
    }

    // Calculate number of output symbols
    U64 inputSamples = input.buffer.size();
    U64 outputSymbols = inputSamples / pimpl->samplesPerSymbol;
    if (outputSymbols == 0) {
        JST_ERROR("Input buffer too small to produce any symbols.");
        return Result::ERROR;
    }

    // Set up output buffer shape - same as input but potentially different size
    std::vector<U64> output_shape = input.buffer.shape();
    output_shape[output_shape.size() - 1] = outputSymbols;  // Last dimension is the sample dimension

    // Allocate output buffer
    output.buffer = Tensor<D, T>(output_shape);

    // Initialize PSK demod state
    pimpl->phaseAccumulator = 0.0;
    pimpl->frequencyError = 0.0;

    F64 nominalOmega = config.sampleRate / config.symbolRate;
    pimpl->timingOmegaNominal = nominalOmega;
    pimpl->timingOmega = nominalOmega;
    pimpl->timingMu = 0.0;
    pimpl->timingIndex = 0;
    pimpl->timingOmegaMin = std::max(0.5, nominalOmega * 0.5);
    pimpl->timingOmegaMax = std::max(pimpl->timingOmegaMin + 1e-6, nominalOmega * 1.5);
    pimpl->sampleHistory.clear();
    pimpl->hasLastSymbol = false;
    pimpl->lastSymbol = T{0};
    pimpl->lastDecision = T{0};

    JST_CHECK(pimpl->refresh_values(config));

    // Get constellation order
    switch (config.pskType) {
        case PskType::BPSK:
            pimpl->constellationOrder = 2;
            break;
        case PskType::QPSK:
            pimpl->constellationOrder = 4;
            break;
        case PskType::PSK8:
            pimpl->constellationOrder = 8;
            break;
        default:
            JST_ERROR("Unsupported PSK type.");
            return Result::ERROR;
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
void PskDemod<D, T>::info() const {
    JST_DEBUG("  PSK Type: {}", config.pskType);
    JST_DEBUG("  Sample Rate: {} Hz", config.sampleRate);
    JST_DEBUG("  Symbol Rate: {} Hz", config.symbolRate);
    JST_DEBUG("  Samples per Symbol: {}", pimpl->samplesPerSymbol);
    JST_DEBUG("  Frequency Loop BW: {}", config.frequencyLoopBandwidth);
    JST_DEBUG("  Timing Loop BW: {}", config.timingLoopBandwidth);
    JST_DEBUG("  Damping Factor: {}", config.dampingFactor);
    JST_DEBUG("  Excess Bandwidth: {}", config.excessBandwidth);
    JST_DEBUG("  Buffer Size: {}", config.bufferSize);
}

template<Device D, typename T>
Result PskDemod<D, T>::setFrequencyLoopBandwidth(F64& frequencyLoopBandwidth) {
    if (frequencyLoopBandwidth <= 0 || frequencyLoopBandwidth >= 1.0) {
        JST_WARN("Frequency loop bandwidth must be between 0 and 1.");
        frequencyLoopBandwidth = config.frequencyLoopBandwidth;
        return Result::WARNING;
    }
    config.frequencyLoopBandwidth = frequencyLoopBandwidth;
    JST_CHECK(pimpl->refresh_values(config));
    return Result::SUCCESS;
}

template<Device D, typename T>
Result PskDemod<D, T>::setTimingLoopBandwidth(F64& timingLoopBandwidth) {
    if (timingLoopBandwidth <= 0 || timingLoopBandwidth >= 1.0) {
        JST_WARN("Timing loop bandwidth must be between 0 and 1.");
        timingLoopBandwidth = config.timingLoopBandwidth;
        return Result::WARNING;
    }
    config.timingLoopBandwidth = timingLoopBandwidth;
    JST_CHECK(pimpl->refresh_values(config));
    return Result::SUCCESS;
}

template<Device D, typename T>
Result PskDemod<D, T>::setDampingFactor(F64& dampingFactor) {
    if (dampingFactor < 0) {
        JST_WARN("Damping factor must be positive.");
        dampingFactor = config.dampingFactor;
        return Result::WARNING;
    }
    config.dampingFactor = dampingFactor;
    JST_CHECK(pimpl->refresh_values(config));
    return Result::SUCCESS;
}

template<Device D, typename T>
Result PskDemod<D, T>::Impl::refresh_values(const Config& config) {
    F64 damp = config.dampingFactor;
    F64 bw = config.frequencyLoopBandwidth;
    F64 denominator = 1.0 + 2.0 * damp * bw + bw * bw;
    freqAlpha = (4.0 * damp * bw) / denominator;
    freqBeta = (4.0 * bw * bw) / denominator;

    bw = config.timingLoopBandwidth;
    denominator = 1.0 + 2.0 * damp * bw + bw * bw;
    timingAlpha = (4.0 * damp * bw) / denominator;
    timingBeta = (4.0 * bw * bw) / denominator;

    return Result::SUCCESS;
}

}  // namespace Jetstream
