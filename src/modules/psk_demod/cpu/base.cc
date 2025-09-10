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

    // Initialize delay line for interpolation (need at least 4 samples)
    pimpl->delayLine.resize(4, T{0});
    pimpl->delayLineIndex = 0;

    return Result::SUCCESS;
}

template<Device D, typename T>
void PskDemod<D, T>::Impl::initializeParameters() {
    // Initialize state variables to zero
    timingAccumulator = 0.0;
    timingError = 0.0;
    phaseAccumulator = 0.0;
    frequencyError = 0.0;
    lastSample = T{0};
}

template<Device D, typename T>
T PskDemod<D, T>::Impl::interpolate(const std::vector<T>& samples, F64 mu) const {
    // Cubic interpolation for timing recovery
    if (samples.size() < 4) {
        return samples.empty() ? T{0} : samples.back();
    }

    // Clamp mu to reasonable bounds
    mu = std::max(0.0, std::min(mu, 1.0));

    // Cubic interpolation coefficients
    F64 a = -0.5 * mu * mu * mu + mu * mu - 0.5 * mu;
    F64 b = 1.5 * mu * mu * mu - 2.5 * mu * mu + 1.0;
    F64 c = -1.5 * mu * mu * mu + 2.0 * mu * mu + 0.5 * mu;
    F64 d = 0.5 * mu * mu * mu - 0.5 * mu * mu;

    F32 af = static_cast<F32>(a);
    F32 bf = static_cast<F32>(b);
    F32 cf = static_cast<F32>(c);
    F32 df = static_cast<F32>(d);
    return T(
        af * samples[0] + bf * samples[1] + cf * samples[2] + df * samples[3]
    );
}

template<Device D, typename T>
F64 PskDemod<D, T>::Impl::gardnerTimingError(const T& early, const T& prompt, const T& late) const {
    // Gardner timing error detector
    std::complex<F32> e = early;
    std::complex<F32> p = prompt;
    std::complex<F32> l = late;

    // Real part of (late - early) * conj(prompt)
    F64 error = std::real((l - e) * std::conj(p));

    // Clamp error to reasonable bounds
    return std::max(MIN_TIMING_ERROR, std::min(error, MAX_TIMING_ERROR));
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

    // Early return for empty buffers
    if (inputSize == 0 || outputSize == 0) {
        return Result::SUCCESS;
    }

    U64 inputIndex = 0;
    U64 outputIndex = 0;

    // Add safety counter to prevent infinite loops
    U64 iterations = 0;
    const U64 maxIterations = inputSize * 3; // Safety limit with generous margin

    // Process input samples
    while (inputIndex < inputSize && outputIndex < outputSize && iterations < maxIterations) {
        iterations++;

        T sample = input.buffer[inputIndex];

        // Update delay line for interpolation
        pimpl->delayLine[pimpl->delayLineIndex] = sample;
        pimpl->delayLineIndex = (pimpl->delayLineIndex + 1) % pimpl->delayLine.size();

        // Update timing accumulator
        pimpl->timingAccumulator += 1.0;

        // Check if we should output a symbol
        if (pimpl->timingAccumulator >= static_cast<F64>(pimpl->samplesPerSymbol)) {
            pimpl->timingAccumulator -= static_cast<F64>(pimpl->samplesPerSymbol);

            // Interpolate symbol at optimal timing point
            F64 mu = pimpl->timingAccumulator / static_cast<F64>(pimpl->samplesPerSymbol);
            T interpolatedSample = pimpl->interpolate(pimpl->delayLine, mu);

            // Apply frequency correction to interpolated sample
            T symbol = pimpl->correctFrequency(interpolatedSample, pimpl->phaseAccumulator);

            // Calculate timing error (Gardner detector) using properly spaced samples
            // We need samples spaced by T/2 (half symbol period) for Gardner detector
            if (pimpl->delayLine.size() >= 4) {
                // Use interpolated samples from delay line at proper spacing
                // Get samples at mu-0.5, mu, and mu+0.5 relative positions
                F64 earlyMu = mu - 0.5;
                F64 lateMu = mu + 0.5;

                // Clamp mu values to valid range
                earlyMu = std::max(0.0, std::min(earlyMu, 1.0));
                lateMu = std::max(0.0, std::min(lateMu, 1.0));

                T early = pimpl->interpolate(pimpl->delayLine, earlyMu);
                T late = pimpl->interpolate(pimpl->delayLine, lateMu);
                F64 timingErr = pimpl->gardnerTimingError(early, symbol, late);

                // Update timing loop with bounded error
                pimpl->timingError += pimpl->timingAlpha * timingErr;
                pimpl->timingAccumulator += pimpl->timingBeta * timingErr;

                // Prevent timing accumulator from going negative or too large
                pimpl->timingAccumulator = std::max(0.0,
                    std::min(pimpl->timingAccumulator,
                             static_cast<F64>(pimpl->samplesPerSymbol) * 2.0));
            }

            // Calculate frequency error (Costas loop)
            F64 frequencyErr = pimpl->costasLoopError(symbol);

            // Update frequency/phase loop with bounded error
            pimpl->frequencyError += pimpl->freqAlpha * frequencyErr;
            pimpl->phaseAccumulator += pimpl->frequencyError + pimpl->freqBeta * frequencyErr;

            // Wrap phase accumulator to [-π, π]
            while (pimpl->phaseAccumulator > M_PI) {
                pimpl->phaseAccumulator -= 2.0 * M_PI;
            }
            while (pimpl->phaseAccumulator < -M_PI) {
                pimpl->phaseAccumulator += 2.0 * M_PI;
            }

            // Bound frequency error accumulator to prevent runaway
            pimpl->frequencyError = std::max(-M_PI, std::min(pimpl->frequencyError, M_PI));

            // Output the soft symbol (frequency/phase corrected)
            output.buffer[outputIndex] = symbol;
            outputIndex++;
        }

        inputIndex++;
        pimpl->lastSample = sample;
    }

    // Check if we hit the safety limit
    if (iterations >= maxIterations) {
        JST_TRACE("PSK Demod: Hit maximum iteration limit, possible infinite loop detected");
        // Continue with zero-filling output
    }

    // Zero-fill remaining output if we didn't produce enough symbols
    while (outputIndex < outputSize) {
        output.buffer[outputIndex] = T{0};
        outputIndex++;
    }

    return Result::SUCCESS;
}

JST_PSK_DEMOD_CPU(JST_INSTANTIATION)

}  // namespace Jetstream
