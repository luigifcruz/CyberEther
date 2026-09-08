#include <algorithm>
#include <cmath>
#include <limits>

#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

constexpr F64 kMaxF32 = std::numeric_limits<F32>::max();
const F64 kMaxSafeCF32Magnitude = static_cast<F64>(
    std::nextafter(std::numeric_limits<F32>::max(), 0.0f));

F64 SamplePower(const F32 sample) {
    const F64 value = sample;
    return value * value;
}

F64 SamplePower(const CF32 sample) {
    const F64 real = sample.real();
    const F64 imag = sample.imag();
    return real * real + imag * imag;
}

F64 LimitGainToFiniteRange(const F64 magnitude,
                           const F64 gain,
                           const F64 limit = kMaxF32) {
    return magnitude > limit / gain
        ? std::nextafter(limit / magnitude, 0.0)
        : gain;
}

F32 ClampToF32(const F64 value) {
    return static_cast<F32>(std::clamp(value, -kMaxF32, kMaxF32));
}

F32 ApplyGain(const F32 sample, const F64 gain) {
    const F64 value = sample;
    const F64 safeGain = LimitGainToFiniteRange(std::abs(value), gain);
    return ClampToF32(value * safeGain);
}

CF32 ApplyGain(const CF32 sample, const F64 gain) {
    const F64 real = sample.real();
    const F64 imag = sample.imag();
    const F64 safeGain = LimitGainToFiniteRange(
        std::hypot(real, imag), gain, kMaxSafeCF32Magnitude);
    return {
        ClampToF32(real * safeGain),
        ClampToF32(imag * safeGain),
    };
}

F64 LimitGainChange(const F64 gain,
                    const F64 previousGain,
                    const F64 minGain,
                    const F64 maxGain,
                    const F64 maxGainChange) {
    const F64 minimumAllowed = std::max(
        minGain, previousGain / maxGainChange);
    const F64 maximumAllowed = previousGain > maxGain / maxGainChange
        ? maxGain
        : previousGain * maxGainChange;
    return std::clamp(gain, minimumAllowed, maximumAllowed);
}

template<typename T>
void ApplyTiledRmsAgc(const Tensor& input,
                      Tensor& output,
                      const Index sampleAxis,
                      const U64 laneCount,
                      const U64 tileSize,
                      const F64 reference,
                      const F64 epsilon,
                      const F64 minGain,
                      const F64 maxGain,
                      const F64 maxGainChange) {
    const T* in = input.data<T>();
    T* out = output.data<T>();
    const U64 sampleCount = input.shape(sampleAxis);
    const U64 inputSampleStride = input.stride(sampleAxis);
    const U64 outputSampleStride = output.stride(sampleAxis);

    for (U64 lane = 0; lane < laneCount; ++lane) {
        U64 coordinates = lane;
        U64 inputLaneOffset = 0;
        U64 outputLaneOffset = 0;
        for (Index axis = input.rank(); axis-- > 0;) {
            if (axis == sampleAxis) {
                continue;
            }
            const U64 coordinate = coordinates % input.shape(axis);
            coordinates /= input.shape(axis);
            inputLaneOffset += coordinate * input.stride(axis);
            outputLaneOffset += coordinate * output.stride(axis);
        }

        const U64 tileCount = 1 + (sampleCount - 1) / tileSize;
        const auto calculateGain = [&](const U64 tile) {
            const U64 tileStart = tile * tileSize;
            const U64 tileLength = std::min(tileSize, sampleCount - tileStart);

            F64 powerSum = 0.0;
            for (U64 sample = 0; sample < tileLength; ++sample) {
                powerSum += SamplePower(
                    in[inputLaneOffset +
                       (tileStart + sample) * inputSampleStride]);
            }

            const F64 meanPower = powerSum / static_cast<F64>(tileLength);
            return std::clamp(
                reference / std::sqrt(meanPower + epsilon), minGain, maxGain);
        };

        F64 startGain = calculateGain(0);
        for (U64 tile = 0; tile < tileCount; ++tile) {
            const U64 tileStart = tile * tileSize;
            const U64 tileLength = std::min(tileSize, sampleCount - tileStart);
            const F64 endGain = tile + 1 < tileCount
                ? LimitGainChange(calculateGain(tile + 1), startGain,
                                  minGain, maxGain, maxGainChange)
                : startGain;
            const F64 gainStep =
                (endGain - startGain) / static_cast<F64>(tileLength);

            for (U64 sample = 0; sample < tileLength; ++sample) {
                const U64 sampleIndex = tileStart + sample;
                const F64 gain = startGain + gainStep * static_cast<F64>(sample);
                out[outputLaneOffset + sampleIndex * outputSampleStride] =
                    ApplyGain(
                        in[inputLaneOffset + sampleIndex * inputSampleStride],
                        gain);
            }

            startGain = endGain;
        }
    }
}

}  // namespace

struct AgcImplNativeCpu : public AgcImpl,
                          public NativeCpuRuntimeContext,
                          public Scheduler::Context {
 public:
    Result validate() final;
    Result create() final;

    Result computeSubmit() override;

 private:
    Result kernelCF32();
    Result kernelF32();

    std::function<Result()> kernel;
};

Result AgcImplNativeCpu::validate() {
    JST_CHECK(AgcImpl::validate());

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() == DataType::CF32) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() == DataType::F32) {
        return Result::SUCCESS;
    }

    JST_ERROR("[MODULE_AGC_NATIVE_CPU] Unsupported data type '{}'.", inputTensor.dtype());
    return Result::ERROR;
}

Result AgcImplNativeCpu::create() {
    JST_CHECK(AgcImpl::create());

    if (input.dtype() == DataType::CF32) {
        kernel = [this]() { return kernelCF32(); };
    } else {
        kernel = [this]() { return kernelF32(); };
    }

    return Result::SUCCESS;
}

Result AgcImplNativeCpu::computeSubmit() {
    return kernel();
}

Result AgcImplNativeCpu::kernelCF32() {
    ApplyTiledRmsAgc<CF32>(input, output, sampleAxis, laneCount, tileSize,
                           reference, epsilon, minGain, maxGain, maxGainChange);
    return Result::SUCCESS;
}

Result AgcImplNativeCpu::kernelF32() {
    ApplyTiledRmsAgc<F32>(input, output, sampleAxis, laneCount, tileSize,
                          reference, epsilon, minGain, maxGain, maxGainChange);
    return Result::SUCCESS;
}

JST_REGISTER_MODULE(AgcImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
