#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <utility>

#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/memory/macros.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

F64 WrapPhase(const F64 value, const F64 period) {
    const F64 wrapped = std::fmod(value, period);
    return wrapped < 0.0 ? wrapped + period : wrapped;
}

}  // namespace

struct SignalGeneratorImplNativeCpu : public SignalGeneratorImpl,
                                      public NativeCpuRuntimeContext,
                                      public Scheduler::Context {
 public:
    Result validate() final;
    Result create() final;
    Result reconfigure() final;

    Result computeSubmit() override;

 private:
    Result kernelSineF32();
    Result kernelSineCF32();
    Result kernelCosineF32();
    Result kernelCosineCF32();
    Result kernelSquareF32();
    Result kernelSquareCF32();
    Result kernelSawtoothF32();
    Result kernelSawtoothCF32();
    Result kernelTriangleF32();
    Result kernelTriangleCF32();
    Result kernelNoiseF32();
    Result kernelNoiseCF32();
    Result kernelDcF32();
    Result kernelDcCF32();
    Result kernelChirpF32();
    Result kernelChirpCF32();
    void advancePhase(F64 frequency);
    void advanceChirpPhase();

    std::function<Result()> kernel;

    // State variables
    F64 oscillatorPhase = 0.0;
    F64 chirpTime = 0.0;
    std::mt19937 rng;
    std::normal_distribution<F64> normalDist;
};

Result SignalGeneratorImplNativeCpu::validate() {
    JST_CHECK(SignalGeneratorImpl::validate());

    U64 alignedOutputSize = 0;
    if (!detail::CheckedPageAlignedSize(validatedOutputSizeBytes, alignedOutputSize) ||
        alignedOutputSize > std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_SIGNAL_GENERATOR_NATIVE_CPU] Output allocation size is too large.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result SignalGeneratorImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(SignalGeneratorImpl::create());

    // Initialize random number generator

    oscillatorPhase = WrapPhase(phase, 2.0 * JST_PI);
    chirpTime = 0.0;
    rng = std::mt19937(std::random_device{}());
    normalDist = std::normal_distribution<F64>(0.0, 1.0);

    // Register compute kernel.

    const bool isF32 = signalDataType == "F32";
    const auto selectKernel = [&](std::function<Result()> f32,
                                  std::function<Result()> cf32) {
        kernel = isF32 ? std::move(f32) : std::move(cf32);
    };

    if (signalType == "sine") {
        selectKernel([this]() { return kernelSineF32(); },
                     [this]() { return kernelSineCF32(); });
    } else if (signalType == "cosine") {
        selectKernel([this]() { return kernelCosineF32(); },
                     [this]() { return kernelCosineCF32(); });
    } else if (signalType == "square") {
        selectKernel([this]() { return kernelSquareF32(); },
                     [this]() { return kernelSquareCF32(); });
    } else if (signalType == "sawtooth") {
        selectKernel([this]() { return kernelSawtoothF32(); },
                     [this]() { return kernelSawtoothCF32(); });
    } else if (signalType == "triangle") {
        selectKernel([this]() { return kernelTriangleF32(); },
                     [this]() { return kernelTriangleCF32(); });
    } else if (signalType == "noise") {
        selectKernel([this]() { return kernelNoiseF32(); },
                     [this]() { return kernelNoiseCF32(); });
    } else if (signalType == "dc") {
        selectKernel([this]() { return kernelDcF32(); },
                     [this]() { return kernelDcCF32(); });
    } else {
        selectKernel([this]() { return kernelChirpF32(); },
                     [this]() { return kernelChirpCF32(); });
    }

    return Result::SUCCESS;
}

Result SignalGeneratorImplNativeCpu::reconfigure() {
    const auto& config = *candidate();
    const bool phaseBased = signalType == "sine" ||
                            signalType == "cosine" ||
                            signalType == "square" ||
                            signalType == "triangle" ||
                            signalType == "sawtooth" ||
                            signalType == "chirp";
    const F64 phaseDelta = phaseBased ?
        WrapPhase(config.phase, 2.0 * JST_PI) -
        WrapPhase(phase, 2.0 * JST_PI) : 0.0;
    const F64 chirpFraction = signalType == "chirp" ?
        chirpTime / chirpDuration : 0.0;

    JST_CHECK(SignalGeneratorImpl::reconfigure());

    if (phaseBased) {
        oscillatorPhase = WrapPhase(oscillatorPhase + phaseDelta,
                                    2.0 * JST_PI);
    }
    if (signalType == "chirp") {
        chirpTime = chirpFraction * chirpDuration;
    }

    return Result::SUCCESS;
}

Result SignalGeneratorImplNativeCpu::computeSubmit() {
    return kernel();
}

void SignalGeneratorImplNativeCpu::advancePhase(const F64 currentFrequency) {
    oscillatorPhase = WrapPhase(
        oscillatorPhase + 2.0 * JST_PI * currentFrequency / sampleRate,
        2.0 * JST_PI);
}

void SignalGeneratorImplNativeCpu::advanceChirpPhase() {
    const F64 dt = 1.0 / sampleRate;
    const F64 chirpRate = (chirpEndFreq - chirpStartFreq) / chirpDuration;
    const auto integrate = [&](const F64 start, const F64 duration) {
        return (chirpStartFreq + chirpRate * start) * duration +
               0.5 * chirpRate * duration * duration;
    };

    F64 cycles = 0.0;
    const F64 untilBoundary = chirpDuration - chirpTime;
    if (dt < untilBoundary) {
        cycles = integrate(chirpTime, dt);
        chirpTime += dt;
    } else {
        cycles = integrate(chirpTime, untilBoundary);
        const F64 afterBoundary = dt - untilBoundary;
        chirpTime = afterBoundary;
        if (afterBoundary > 0.0) {
            cycles += integrate(0.0, afterBoundary);
        }
    }
    oscillatorPhase = WrapPhase(oscillatorPhase + 2.0 * JST_PI * cycles,
                                2.0 * JST_PI);
}

Result SignalGeneratorImplNativeCpu::kernelSineF32() {
    return AutomaticIterator<F32>(
        [&](auto& out) {
            const F64 value = amplitude * std::sin(oscillatorPhase) + dcOffset;
            out = static_cast<F32>(value);
            advancePhase(frequency);
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelSineCF32() {
    return AutomaticIterator<CF32>(
        [&](auto& out) {
            const F64 iValue = amplitude * std::sin(oscillatorPhase) +
                               dcOffset;
            const F64 qValue = -amplitude * std::cos(oscillatorPhase);
            out = CF32(static_cast<F32>(iValue), static_cast<F32>(qValue));
            advancePhase(frequency);
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelCosineF32() {
    return AutomaticIterator<F32>(
        [&](auto& out) {
            const F64 value = amplitude * std::cos(oscillatorPhase) + dcOffset;
            out = static_cast<F32>(value);
            advancePhase(frequency);
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelCosineCF32() {
    return AutomaticIterator<CF32>(
        [&](auto& out) {
            const F64 iVal = amplitude * std::cos(oscillatorPhase) + dcOffset;
            const F64 qVal = amplitude * std::sin(oscillatorPhase);
            out = CF32(static_cast<F32>(iVal), static_cast<F32>(qVal));
            advancePhase(frequency);
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelSquareF32() {
    return AutomaticIterator<F32>(
        [&](auto& out) {
            const F64 value = amplitude *
                (oscillatorPhase < JST_PI ? 1.0 : -1.0) + dcOffset;
            out = static_cast<F32>(value);
            advancePhase(frequency);
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelSquareCF32() {
    return AutomaticIterator<CF32>(
        [&](auto& out) {
            const F64 value = amplitude *
                (oscillatorPhase < JST_PI ? 1.0 : -1.0) + dcOffset;
            out = CF32(static_cast<F32>(value), 0.0f);
            advancePhase(frequency);
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelSawtoothF32() {
    return AutomaticIterator<F32>(
        [&](auto& out) {
            const F64 phaseVal = oscillatorPhase / (2.0 * JST_PI);
            const F64 value = amplitude * (2.0 * phaseVal - 1.0) + dcOffset;
            out = static_cast<F32>(value);
            advancePhase(frequency);
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelSawtoothCF32() {
    return AutomaticIterator<CF32>(
        [&](auto& out) {
            const F64 phaseVal = oscillatorPhase / (2.0 * JST_PI);
            const F64 value = amplitude * (2.0 * phaseVal - 1.0) + dcOffset;
            out = CF32(static_cast<F32>(value), 0.0f);
            advancePhase(frequency);
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelTriangleF32() {
    return AutomaticIterator<F32>(
        [&](auto& out) {
            const F64 phaseVal = oscillatorPhase / (2.0 * JST_PI);
            const F64 value = amplitude * (phaseVal < 0.5 ?
                4.0 * phaseVal - 1.0 : 3.0 - 4.0 * phaseVal) + dcOffset;
            out = static_cast<F32>(value);
            advancePhase(frequency);
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelTriangleCF32() {
    return AutomaticIterator<CF32>(
        [&](auto& out) {
            const F64 phaseVal = oscillatorPhase / (2.0 * JST_PI);
            const F64 value = amplitude * (phaseVal < 0.5 ?
                4.0 * phaseVal - 1.0 : 3.0 - 4.0 * phaseVal) + dcOffset;
            out = CF32(static_cast<F32>(value), 0.0f);
            advancePhase(frequency);
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelNoiseF32() {
    const F64 scale = amplitude * std::sqrt(noiseVariance);
    constexpr F64 maxF32 = std::numeric_limits<F32>::max();
    return AutomaticIterator<F32>(
        [&](auto& out) {
            const F64 noise = noiseVariance > 0.0 ? normalDist(rng) : 0.0;
            const F64 value = std::clamp(scale * noise + dcOffset,
                                         -maxF32, maxF32);
            out = static_cast<F32>(value);
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelNoiseCF32() {
    const F64 scale = amplitude * std::sqrt(noiseVariance);
    constexpr F64 maxF32 = std::numeric_limits<F32>::max();
    return AutomaticIterator<CF32>(
        [&](auto& out) {
            const F64 iNoise = noiseVariance > 0.0 ? normalDist(rng) : 0.0;
            const F64 qNoise = noiseVariance > 0.0 ? normalDist(rng) : 0.0;
            const F64 iVal = std::clamp(scale * iNoise + dcOffset,
                                        -maxF32, maxF32);
            const F64 qVal = std::clamp(scale * qNoise, -maxF32, maxF32);
            out = CF32(static_cast<F32>(iVal), static_cast<F32>(qVal));
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelDcF32() {
    const F64 dcValue = amplitude + dcOffset;

    return AutomaticIterator<F32>(
        [&](auto& out) {
            out = static_cast<F32>(dcValue);
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelDcCF32() {
    const F64 dcValue = amplitude + dcOffset;

    return AutomaticIterator<CF32>(
        [&](auto& out) {
            out = CF32(static_cast<F32>(dcValue), 0.0f);
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelChirpF32() {
    return AutomaticIterator<F32>(
        [&](auto& out) {
            const F64 value = amplitude * std::cos(oscillatorPhase) + dcOffset;
            out = static_cast<F32>(value);
            advanceChirpPhase();
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelChirpCF32() {
    return AutomaticIterator<CF32>(
        [&](auto& out) {
            const F64 iVal = amplitude * std::cos(oscillatorPhase) + dcOffset;
            const F64 qVal = amplitude * std::sin(oscillatorPhase);
            out = CF32(static_cast<F32>(iVal), static_cast<F32>(qVal));
            advanceChirpPhase();
        },
        signal);
}

JST_REGISTER_MODULE(SignalGeneratorImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
