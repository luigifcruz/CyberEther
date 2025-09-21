#include "../generic.cc"

#include <cmath>
#include <complex>
#include <algorithm>

namespace Jetstream {

template<Device D, typename T>
PskDemod<D, T>::PskDemod() {
    pimpl = std::make_unique<Impl>();
}

template<Device D, typename T>
PskDemod<D, T>::~PskDemod() {
    pimpl.reset();
}

template<Device D, typename T>
Result PskDemod<D, T>::createCompute(const Context&) {
    JST_TRACE("Create PSK Demod compute core using CPU backend.");

    // Initialize all parameters
    pimpl->initializeParameters();

    // Reset raw sample history used for interpolation across buffers
    pimpl->sampleHistory.clear();

    return Result::SUCCESS;
}

template<Device D, typename T>
void PskDemod<D, T>::Impl::initializeParameters() {
    // Initialize state variables to zero
    phaseAccumulator = 0.0;
    frequencyError = 0.0;
    timingMu = 0.0;
    timingOmega = timingOmegaNominal;
    timingIndex = 0;
    hasLastSymbol = false;
    lastSymbol = T{0};
    lastDecision = T{0};
    sampleHistory.clear();
}

template<Device D, typename T>
T PskDemod<D, T>::Impl::interpolate(const T& a, const T& b, F64 mu) const {
    mu = std::clamp(mu, 0.0, 1.0);
    F32 frac = static_cast<F32>(mu);
    F32 inv = 1.0f - frac;
    return a * inv + b * frac;
}

template<Device D, typename T>
T PskDemod<D, T>::Impl::decision(const T& sample) const {
    constexpr F32 INV_SQRT2 = 0.7071067811865475f;
    constexpr F32 PI_F = 3.14159265358979323846f;
    constexpr F32 STEP_8PSK = PI_F / 4.0f;

    std::complex<F32> s = sample;

    switch (constellationOrder) {
        case 2: {  // BPSK maps to the real axis
            F32 sign = s.real() >= 0.0f ? 1.0f : -1.0f;
            return T(sign, 0.0f);
        }
        case 4: {  // QPSK aligns to the quadrants
            F32 re = s.real() >= 0.0f ? INV_SQRT2 : -INV_SQRT2;
            F32 im = s.imag() >= 0.0f ? INV_SQRT2 : -INV_SQRT2;
            return T(re, im);
        }
        case 8: {  // 8-PSK selects the nearest constellation point
            F32 phase = std::arg(s);
            F32 decision_phase = std::round(phase / STEP_8PSK) * STEP_8PSK;
            return T(std::polar(1.0f, decision_phase));
        }
        default:
            return sample;
    }
}

template<Device D, typename T>
F64 PskDemod<D, T>::Impl::muellerMullerError(const T& prevSymbol, const T& prevDecision,
                                             const T& currentSymbol, const T& currentDecision) const {
    std::complex<F32> prev_s = prevSymbol;
    std::complex<F32> prev_d = prevDecision;
    std::complex<F32> curr_s = currentSymbol;
    std::complex<F32> curr_d = currentDecision;

    std::complex<F32> term1 = prev_d * std::conj(curr_s);
    std::complex<F32> term2 = prev_s * std::conj(curr_d);

    return static_cast<F64>(std::real(term1 - term2));
}

template<Device D, typename T>
F64 PskDemod<D, T>::Impl::costasLoopError(const T& sample) const {
    std::complex<F32> s = sample;

    F64 error = 0.0;
    switch (constellationOrder) {
        case 2: // BPSK
            error = std::imag(s) * (s.real() > 0 ? 1.0f : -1.0f);
            break;

        case 4: // QPSK
            {
                F32 re = s.real();
                F32 im = s.imag();
                F32 re_sign = re > 0 ? 1.0f : -1.0f;
                F32 im_sign = im > 0 ? 1.0f : -1.0f;
                std::complex<F32> decision(re_sign, im_sign);
                error = std::imag(s * std::conj(decision));
            }
            break;

        case 8: // 8-PSK
            {
                F32 phase = std::arg(s);
                F32 decision_phase = std::round(phase * 4.0f / M_PI) * M_PI / 4.0f;
                error = std::sin(phase - decision_phase);
            }
            break;

        default:
            error = 0.0;
            break;
    }

    // Clamp frequency error to reasonable bounds
    return std::max(MIN_FREQUENCY_ERROR, std::min(error, MAX_FREQUENCY_ERROR));
}

template<Device D, typename T>
T PskDemod<D, T>::Impl::correctFrequency(const T& sample, F64 phase) const {
    std::complex<F32> correction = std::polar(1.0f, static_cast<F32>(-phase));
    return sample * correction;
}

template<Device D, typename T>
Result PskDemod<D, T>::compute(const Context&) {
    const U64 inputSize = input.buffer.size();
    const U64 outputSize = output.buffer.size();

    // Early return for empty output buffers
    if (outputSize == 0) {
        return Result::SUCCESS;
    }

    auto& impl = *pimpl;

    // Append newly received samples to the interpolation history
    for (U64 i = 0; i < inputSize; ++i) {
        impl.sampleHistory.push_back(input.buffer[i]);
    }

    U64 outputIndex = 0;

    // Local copies of the loop state for better cache behaviour
    F64 mu = impl.timingMu;
    F64 omega = impl.timingOmega;
    U64 index = impl.timingIndex;
    F64 phase = impl.phaseAccumulator;
    F64 freqAcc = impl.frequencyError;
    bool hasPrevSymbol = impl.hasLastSymbol;
    T prevSymbol = impl.lastSymbol;
    T prevDecision = impl.lastDecision;

    // Safety counter prevents infinite loops if configuration is pathological
    U64 iterations = 0;
    const U64 maxIterations = outputSize * (impl.samplesPerSymbol + 4);

    while (outputIndex < outputSize && iterations < maxIterations) {
        iterations++;

        const U64 historySize = static_cast<U64>(impl.sampleHistory.size());

        // Bring mu back into [0, 1) while staying within the available history
        while (mu >= 1.0 && index + 1 < historySize) {
            mu -= 1.0;
            ++index;
        }
        while (mu < 0.0 && index > 0) {
            mu += 1.0;
            --index;
        }
        if (mu < 0.0) {
            mu = 0.0;
        }

        // Not enough samples yet to interpolate the next symbol
        if (index + 1 >= historySize) {
            break;
        }

        const T& earlySample = impl.sampleHistory[index];
        const T& lateSample = impl.sampleHistory[index + 1];
        T interpolated = impl.interpolate(earlySample, lateSample, mu);
        T corrected = impl.correctFrequency(interpolated, phase);
        T decision = impl.decision(corrected);

        F64 timingErr = 0.0;
        if (hasPrevSymbol) {
            timingErr = impl.muellerMullerError(prevSymbol, prevDecision, corrected, decision);
            timingErr = std::clamp(timingErr, Impl::MIN_TIMING_ERROR, Impl::MAX_TIMING_ERROR);
            omega += impl.timingBeta * timingErr;
            omega = std::clamp(omega, impl.timingOmegaMin, impl.timingOmegaMax);
            mu += impl.timingAlpha * timingErr;
        }

        F64 freqErrSample = impl.costasLoopError(corrected);
        freqAcc += impl.freqAlpha * freqErrSample;
        freqAcc = std::clamp(freqAcc, -M_PI, M_PI);
        phase += freqAcc + impl.freqBeta * freqErrSample;
        phase = std::remainder(phase, 2.0 * M_PI);

        output.buffer[outputIndex++] = corrected;

        prevSymbol = corrected;
        prevDecision = decision;
        hasPrevSymbol = true;

        mu += omega;
    }

    if (iterations >= maxIterations) {
        JST_TRACE("PSK Demod: Hit maximum iteration limit during MM clock recovery");
    }

    // Discard the samples that are no longer needed while keeping one look-back sample
    std::size_t historySize = impl.sampleHistory.size();
    std::size_t pruneCount = 0;
    if (historySize > 1) {
        pruneCount = std::min<std::size_t>(static_cast<std::size_t>(index), historySize - 1);
        for (std::size_t i = 0; i < pruneCount; ++i) {
            impl.sampleHistory.pop_front();
        }
    }
    if (pruneCount > 0) {
        index -= static_cast<U64>(pruneCount);
    }

    impl.timingMu = mu;
    impl.timingOmega = omega;
    impl.timingIndex = index;
    impl.phaseAccumulator = phase;
    impl.frequencyError = freqAcc;
    impl.hasLastSymbol = hasPrevSymbol;
    impl.lastSymbol = prevSymbol;
    impl.lastDecision = prevDecision;

    // Zero-fill any remaining output slots to preserve deterministic output sizes
    while (outputIndex < outputSize) {
        output.buffer[outputIndex++] = T{0};
    }

    return Result::SUCCESS;
}

JST_PSK_DEMOD_CPU(JST_INSTANTIATION)

}  // namespace Jetstream
