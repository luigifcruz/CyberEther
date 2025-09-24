#include "../generic.cc"

#include "jetstream/macros.hh"
#include "jetstream/memory2/helpers.hh"
#include <random>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Jetstream {

template<Device D, typename T>
struct SignalGenerator<D, T>::Impl {
    U64 sampleIndex = 0;
    std::mt19937 rng;
    std::normal_distribution<F64> normalDist;

    Impl() : rng(std::random_device{}()), normalDist(0.0, 1.0) {}
};

template<Device D, typename T>
SignalGenerator<D, T>::SignalGenerator() {
    pimpl = std::make_unique<Impl>();
}

template<Device D, typename T>
SignalGenerator<D, T>::~SignalGenerator() {
    pimpl.reset();
}

template<Device D, typename T>
Result SignalGenerator<D, T>::createCompute(const Context&) {
    JST_TRACE("Create SignalGenerator compute core using CPU backend.");

    // Reset state
    pimpl->sampleIndex = 0;
    pimpl->normalDist = std::normal_distribution<F64>(0.0, std::sqrt(config.noiseVariance));

    return Result::SUCCESS;
}

template<Device D, typename T>
Result SignalGenerator<D, T>::compute(const Context&) {
    const F64 dt = 1.0 / config.sampleRate;
    const U64 bufferSize = config.bufferSize;

    // Switch once per buffer, not per sample
    switch (config.signalType) {
        case SignalType::Sine: {
            U64 idx = 0;
            mem2::AutomaticIterator([&](auto& out) {
                const F64 t = (pimpl->sampleIndex + idx) * dt;
                const F64 iValue = config.amplitude * std::sin(2.0 * M_PI * config.frequency * t + config.phase) + config.dcOffset;

                if constexpr (std::is_same_v<T, CF32>) {
                    out = CF32(static_cast<F32>(iValue), 0.0f);
                } else {
                    out = static_cast<T>(iValue);
                }
                ++idx;
            }, output.buffer);
            break;
        }

        case SignalType::Cosine: {
            U64 idx = 0;
            mem2::AutomaticIterator([&](auto& out) {
                const F64 t = (pimpl->sampleIndex + idx) * dt;
                const F64 iValue = config.amplitude * std::cos(2.0 * M_PI * config.frequency * t + config.phase) + config.dcOffset;
                const F64 qValue = config.amplitude * std::sin(2.0 * M_PI * config.frequency * t + config.phase);

                if constexpr (std::is_same_v<T, CF32>) {
                    out = CF32(static_cast<F32>(iValue), static_cast<F32>(qValue));
                } else {
                    out = static_cast<T>(iValue);
                }
                ++idx;
            }, output.buffer);
            break;
        }

        case SignalType::Square: {
            U64 idx = 0;
            mem2::AutomaticIterator([&](auto& out) {
                const F64 t = (pimpl->sampleIndex + idx) * dt;
                const F64 phase = std::fmod(2.0 * M_PI * config.frequency * t + config.phase, 2.0 * M_PI);
                const F64 iValue = config.amplitude * ((phase < M_PI) ? 1.0 : -1.0) + config.dcOffset;

                if constexpr (std::is_same_v<T, CF32>) {
                    out = CF32(static_cast<F32>(iValue), 0.0f);
                } else {
                    out = static_cast<T>(iValue);
                }
                ++idx;
            }, output.buffer);
            break;
        }

        case SignalType::Sawtooth: {
            U64 idx = 0;
            mem2::AutomaticIterator([&](auto& out) {
                const F64 t = (pimpl->sampleIndex + idx) * dt;
                const F64 phase = std::fmod(config.frequency * t + config.phase / (2.0 * M_PI), 1.0);
                const F64 iValue = config.amplitude * (2.0 * phase - 1.0) + config.dcOffset;

                if constexpr (std::is_same_v<T, CF32>) {
                    out = CF32(static_cast<F32>(iValue), 0.0f);
                } else {
                    out = static_cast<T>(iValue);
                }
                ++idx;
            }, output.buffer);
            break;
        }

        case SignalType::Triangle: {
            U64 idx = 0;
            mem2::AutomaticIterator([&](auto& out) {
                const F64 t = (pimpl->sampleIndex + idx) * dt;
                const F64 phase = std::fmod(config.frequency * t + config.phase / (2.0 * M_PI), 1.0);
                const F64 iValue = config.amplitude * ((phase < 0.5) ? (4.0 * phase - 1.0) : (3.0 - 4.0 * phase)) + config.dcOffset;

                if constexpr (std::is_same_v<T, CF32>) {
                    out = CF32(static_cast<F32>(iValue), 0.0f);
                } else {
                    out = static_cast<T>(iValue);
                }
                ++idx;
            }, output.buffer);
            break;
        }

        case SignalType::Noise: {
            mem2::AutomaticIterator([&](auto& out) {
                const F64 iValue = config.amplitude * pimpl->normalDist(pimpl->rng) + config.dcOffset;

                if constexpr (std::is_same_v<T, CF32>) {
                    const F64 qValue = config.amplitude * pimpl->normalDist(pimpl->rng);
                    out = CF32(static_cast<F32>(iValue), static_cast<F32>(qValue));
                } else {
                    out = static_cast<T>(iValue);
                }
            }, output.buffer);
            break;
        }

        case SignalType::DC: {
            const F64 dcValue = config.amplitude + config.dcOffset;
            mem2::AutomaticIterator([&](auto& out) {
                if constexpr (std::is_same_v<T, CF32>) {
                    out = CF32(static_cast<F32>(dcValue), 0.0f);
                } else {
                    out = static_cast<T>(dcValue);
                }
            }, output.buffer);
            break;
        }

        case SignalType::Chirp: {
            U64 idx = 0;
            mem2::AutomaticIterator([&](auto& out) {
                const F64 t = (pimpl->sampleIndex + idx) * dt;
                const F64 chirpTime = std::fmod(t, config.chirpDuration);
                const F64 chirpRate = (config.chirpEndFreq - config.chirpStartFreq) / config.chirpDuration;
                const F64 chirpPhase = 2.0 * M_PI * (config.chirpStartFreq * chirpTime + 0.5 * chirpRate * chirpTime * chirpTime);
                const F64 iValue = config.amplitude * std::cos(chirpPhase + config.phase) + config.dcOffset;

                if constexpr (std::is_same_v<T, CF32>) {
                    const F64 qValue = config.amplitude * std::sin(chirpPhase + config.phase);
                    out = CF32(static_cast<F32>(iValue), static_cast<F32>(qValue));
                } else {
                    out = static_cast<T>(iValue);
                }
                ++idx;
            }, output.buffer);
            break;
        }
    }

    // Update sample index
    pimpl->sampleIndex += bufferSize;

    return Result::SUCCESS;
}

JST_SIGNAL_GENERATOR_CPU(JST_INSTANTIATION)
JST_SIGNAL_GENERATOR_CPU(JST_BENCHMARK)

}  // namespace Jetstream
