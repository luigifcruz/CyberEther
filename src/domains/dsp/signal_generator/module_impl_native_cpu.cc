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

    std::function<Result()> kernel;

    // State variables
    U64 sampleIndex = 0;
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

    sampleIndex = 0;
    rng = std::mt19937(std::random_device{}());
    normalDist = std::normal_distribution<F64>();
    if (noiseVariance > 0.0) {
        normalDist = std::normal_distribution<F64>(0.0, std::sqrt(noiseVariance));
    }

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

Result SignalGeneratorImplNativeCpu::computeSubmit() {
    auto result = kernel();
    sampleIndex += bufferSize;
    return result;
}

Result SignalGeneratorImplNativeCpu::kernelSineF32() {
    const F64 dt = 1.0 / sampleRate;
    U64 idx = 0;

    return AutomaticIterator<F32>(
        [&](auto& out) {
            const F64 t = (sampleIndex + idx) * dt;
            const F64 value = amplitude * std::sin(2.0 * JST_PI * frequency * t + phase) + dcOffset;
            out = static_cast<F32>(value);
            ++idx;
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelSineCF32() {
    const F64 dt = 1.0 / sampleRate;
    U64 idx = 0;

    return AutomaticIterator<CF32>(
        [&](auto& out) {
            const F64 t = (sampleIndex + idx) * dt;
            const F64 value = amplitude * std::sin(2.0 * JST_PI * frequency * t + phase) + dcOffset;
            out = CF32(static_cast<F32>(value), 0.0f);
            ++idx;
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelCosineF32() {
    const F64 dt = 1.0 / sampleRate;
    U64 idx = 0;

    return AutomaticIterator<F32>(
        [&](auto& out) {
            const F64 t = (sampleIndex + idx) * dt;
            const F64 value = amplitude * std::cos(2.0 * JST_PI * frequency * t + phase) + dcOffset;
            out = static_cast<F32>(value);
            ++idx;
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelCosineCF32() {
    const F64 dt = 1.0 / sampleRate;
    U64 idx = 0;

    return AutomaticIterator<CF32>(
        [&](auto& out) {
            const F64 t = (sampleIndex + idx) * dt;
            const F64 iVal = amplitude * std::cos(2.0 * JST_PI * frequency * t + phase) + dcOffset;
            const F64 qVal = amplitude * std::sin(2.0 * JST_PI * frequency * t + phase);
            out = CF32(static_cast<F32>(iVal), static_cast<F32>(qVal));
            ++idx;
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelSquareF32() {
    const F64 dt = 1.0 / sampleRate;
    U64 idx = 0;

    return AutomaticIterator<F32>(
        [&](auto& out) {
            const F64 t = (sampleIndex + idx) * dt;
            const F64 phaseVal = WrapPhase(2.0 * JST_PI * frequency * t + phase,
                                           2.0 * JST_PI);
            const F64 value = amplitude * ((phaseVal < JST_PI) ? 1.0 : -1.0) + dcOffset;
            out = static_cast<F32>(value);
            ++idx;
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelSquareCF32() {
    const F64 dt = 1.0 / sampleRate;
    U64 idx = 0;

    return AutomaticIterator<CF32>(
        [&](auto& out) {
            const F64 t = (sampleIndex + idx) * dt;
            const F64 phaseVal = WrapPhase(2.0 * JST_PI * frequency * t + phase,
                                           2.0 * JST_PI);
            const F64 value = amplitude * ((phaseVal < JST_PI) ? 1.0 : -1.0) + dcOffset;
            out = CF32(static_cast<F32>(value), 0.0f);
            ++idx;
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelSawtoothF32() {
    const F64 dt = 1.0 / sampleRate;
    U64 idx = 0;

    return AutomaticIterator<F32>(
        [&](auto& out) {
            const F64 t = (sampleIndex + idx) * dt;
            const F64 phaseVal = WrapPhase(frequency * t + phase / (2.0 * JST_PI), 1.0);
            const F64 value = amplitude * (2.0 * phaseVal - 1.0) + dcOffset;
            out = static_cast<F32>(value);
            ++idx;
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelSawtoothCF32() {
    const F64 dt = 1.0 / sampleRate;
    U64 idx = 0;

    return AutomaticIterator<CF32>(
        [&](auto& out) {
            const F64 t = (sampleIndex + idx) * dt;
            const F64 phaseVal = WrapPhase(frequency * t + phase / (2.0 * JST_PI), 1.0);
            const F64 value = amplitude * (2.0 * phaseVal - 1.0) + dcOffset;
            out = CF32(static_cast<F32>(value), 0.0f);
            ++idx;
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelTriangleF32() {
    const F64 dt = 1.0 / sampleRate;
    U64 idx = 0;

    return AutomaticIterator<F32>(
        [&](auto& out) {
            const F64 t = (sampleIndex + idx) * dt;
            const F64 phaseVal = WrapPhase(frequency * t + phase / (2.0 * JST_PI), 1.0);
            const F64 value = amplitude * ((phaseVal < 0.5) ? (4.0 * phaseVal - 1.0) : (3.0 - 4.0 * phaseVal)) + dcOffset;
            out = static_cast<F32>(value);
            ++idx;
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelTriangleCF32() {
    const F64 dt = 1.0 / sampleRate;
    U64 idx = 0;

    return AutomaticIterator<CF32>(
        [&](auto& out) {
            const F64 t = (sampleIndex + idx) * dt;
            const F64 phaseVal = WrapPhase(frequency * t + phase / (2.0 * JST_PI), 1.0);
            const F64 value = amplitude * ((phaseVal < 0.5) ? (4.0 * phaseVal - 1.0) : (3.0 - 4.0 * phaseVal)) + dcOffset;
            out = CF32(static_cast<F32>(value), 0.0f);
            ++idx;
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelNoiseF32() {
    return AutomaticIterator<F32>(
        [&](auto& out) {
            const F64 noise = noiseVariance > 0.0 ? normalDist(rng) : 0.0;
            const F64 value = amplitude * noise + dcOffset;
            out = static_cast<F32>(value);
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelNoiseCF32() {
    return AutomaticIterator<CF32>(
        [&](auto& out) {
            const F64 iNoise = noiseVariance > 0.0 ? normalDist(rng) : 0.0;
            const F64 qNoise = noiseVariance > 0.0 ? normalDist(rng) : 0.0;
            const F64 iVal = amplitude * iNoise + dcOffset;
            const F64 qVal = amplitude * qNoise;
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
    const F64 dt = 1.0 / sampleRate;
    U64 idx = 0;

    return AutomaticIterator<F32>(
        [&](auto& out) {
            const F64 t = (sampleIndex + idx) * dt;
            const F64 chirpTime = std::fmod(t, chirpDuration);
            const F64 chirpRate = (chirpEndFreq - chirpStartFreq) / chirpDuration;
            const F64 chirpPhase = 2.0 * JST_PI * (chirpStartFreq * chirpTime + 0.5 * chirpRate * chirpTime * chirpTime);
            const F64 value = amplitude * std::cos(chirpPhase + phase) + dcOffset;
            out = static_cast<F32>(value);
            ++idx;
        },
        signal);
}

Result SignalGeneratorImplNativeCpu::kernelChirpCF32() {
    const F64 dt = 1.0 / sampleRate;
    U64 idx = 0;

    return AutomaticIterator<CF32>(
        [&](auto& out) {
            const F64 t = (sampleIndex + idx) * dt;
            const F64 chirpTime = std::fmod(t, chirpDuration);
            const F64 chirpRate = (chirpEndFreq - chirpStartFreq) / chirpDuration;
            const F64 chirpPhase = 2.0 * JST_PI * (chirpStartFreq * chirpTime + 0.5 * chirpRate * chirpTime * chirpTime);
            const F64 iVal = amplitude * std::cos(chirpPhase + phase) + dcOffset;
            const F64 qVal = amplitude * std::sin(chirpPhase + phase);
            out = CF32(static_cast<F32>(iVal), static_cast<F32>(qVal));
            ++idx;
        },
        signal);
}

JST_REGISTER_MODULE(SignalGeneratorImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
