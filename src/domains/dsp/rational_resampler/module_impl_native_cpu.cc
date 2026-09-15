#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <jetstream/memory/macros.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/tools/numeric.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct RationalResamplerImplNativeCpu : public RationalResamplerImpl,
                                        public NativeCpuRuntimeContext,
                                        public Scheduler::Context {
    Result validate() final;
    Result create() final;
    Result computeSubmit() override;

 private:
    Result generateCoefficients();

    template<typename T>
    Result resampleKernel();

    Tensor coefficients;
    Tensor history;
    Tensor workspace;
    Tensor pending;
    U64 nextTime = 0;
    U64 pendingCount = 0;
};

Result RationalResamplerImplNativeCpu::validate() {
    JST_CHECK(RationalResamplerImpl::validate());
    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }
    const Tensor& tensor = inputs().at("buffer").tensor;
    if (!tensor.validShape() || tensor.size() == 0) {
        return Result::SUCCESS;
    }
    if (tensor.dtype() != DataType::F32 && tensor.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_RATIONAL_RESAMPLER_NATIVE_CPU] Input must be F32 or CF32.");
        return Result::ERROR;
    }

    const auto fitsAllocation = [](const U64 rows, const U64 columns,
                                   const U64 elementSize) {
        U64 bytes = 0;
        U64 alignedBytes = 0;
        return detail::CheckedMultiply(rows, columns, bytes) &&
               detail::CheckedMultiply(bytes, elementSize, bytes) &&
               detail::CheckedPageAlignedSize(bytes, alignedBytes) &&
               alignedBytes <= std::numeric_limits<std::size_t>::max();
    };
    const auto& p = validatedPlan;
    const U64 elementSize = DataTypeSize(tensor.dtype());
    if (!fitsAllocation(p.interpolation, p.phaseLength, sizeof(F32)) ||
        !fitsAllocation(p.laneCount, p.phaseLength - 1, elementSize) ||
        !fitsAllocation(1, p.workspaceSize, elementSize) ||
        !fitsAllocation(p.laneCount, p.pendingCapacity, elementSize) ||
        !fitsAllocation(p.laneCount, p.outputPerLane, elementSize)) {
        JST_ERROR("[MODULE_RATIONAL_RESAMPLER_NATIVE_CPU] Buffers exceed the allocation range.");
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

Result RationalResamplerImplNativeCpu::create() {
    JST_CHECK(RationalResamplerImpl::create());
    JST_CHECK(coefficients.create(input.device(), DataType::F32,
                                   {plan.interpolation, plan.phaseLength}));
    JST_CHECK(history.create(input.device(), input.dtype(),
                              {plan.laneCount, plan.phaseLength - 1}));
    JST_CHECK(workspace.create(input.device(), input.dtype(), {plan.workspaceSize}));
    JST_CHECK(pending.create(input.device(), input.dtype(),
                              {plan.laneCount, plan.pendingCapacity}));
    std::memset(history.data(), 0, history.size() * history.elementSize());
    nextTime = 0;
    pendingCount = 0;
    return generateCoefficients();
}

Result RationalResamplerImplNativeCpu::generateCoefficients() {
    const F64 frequency = 0.5 * cutoff /
                          std::max(plan.interpolation, plan.decimation);
    const F64 center = static_cast<F64>(plan.tapCount - 1) / 2.0;
    F32* data = coefficients.data<F32>();
    std::fill_n(data, coefficients.size(), 0.0f);
    F64 sum = 0.0;

    for (U64 tap = 0; tap < plan.tapCount; ++tap) {
        const F64 time = static_cast<F64>(tap) - center;
        // The common 2*frequency factor cancels during normalization. Omitting
        // it also keeps very small cutoffs representable in the F32 tap bank.
        const F64 argument = 2.0 * JST_PI * frequency * time;
        const F64 sinc = argument == 0.0 ? 1.0 : std::sin(argument) / argument;
        const F64 angle = 2.0 * JST_PI * static_cast<F64>(tap) /
                          static_cast<F64>(plan.tapCount - 1);
        const F64 window = 0.42 - 0.5 * std::cos(angle) + 0.08 * std::cos(2.0 * angle);
        const F64 value = sinc * window;
        // Reverse each phase so both operands of the dot product are contiguous.
        const U64 phase = tap % plan.interpolation;
        const U64 index = plan.phaseLength - 1 - tap / plan.interpolation;
        data[phase * plan.phaseLength + index] = static_cast<F32>(value);
        sum += value;
    }
    const F64 gain = static_cast<F64>(plan.interpolation) / sum;
    for (U64 tap = 0; tap < coefficients.size(); ++tap) {
        data[tap] = static_cast<F32>(data[tap] * gain);
    }
    return Result::SUCCESS;
}

Result RationalResamplerImplNativeCpu::computeSubmit() {
    JST_CHECK(updateSampleRate());
    return input.dtype() == DataType::CF32 ? resampleKernel<CF32>()
                                           : resampleKernel<F32>();
}

template<typename T>
Result RationalResamplerImplNativeCpu::resampleKernel() {
    const Index sampleAxis = *plan.axes.sample;
    const U64 inputSampleStride = input.stride(sampleAxis);
    const U64 outputSampleStride = output.stride(sampleAxis);
    const U64 inputBatchStride = plan.axes.batch ? input.stride(*plan.axes.batch) : 0;
    const U64 outputBatchStride = plan.axes.batch ? output.stride(*plan.axes.batch) : 0;
    const U64 historySize = plan.phaseLength - 1;
    T* window = workspace.data<T>();
    U64 finalTime = nextTime;
    U64 finalCount = pendingCount;

    for (U64 lane = 0; lane < plan.laneCount; ++lane) {
        U64 coordinates = lane;
        U64 inputOffset = 0;
        U64 outputOffset = 0;
        for (Index axis = input.rank(); axis-- > 0;) {
            if (axis == sampleAxis || (plan.axes.batch && axis == *plan.axes.batch)) {
                continue;
            }
            const U64 coordinate = coordinates % input.shape(axis);
            coordinates /= input.shape(axis);
            inputOffset += coordinate * input.stride(axis);
            outputOffset += coordinate * output.stride(axis);
        }

        T* laneHistory = history.data<T>() + lane * historySize;
        T* queued = pending.data<T>() + lane * plan.pendingCapacity;
        U64 time = nextTime;
        U64 count = pendingCount;
        for (U64 batch = 0; batch < plan.batchCount; ++batch) {
            const T* source = input.data<T>() + inputOffset + batch * inputBatchStride;
            std::copy_n(laneHistory, historySize, window);
            for (U64 sample = 0; sample < plan.inputSamples; ++sample) {
                window[historySize + sample] = source[sample * inputSampleStride];
            }

            // Time is in interpolated-rate ticks, relative to this input batch.
            while (time < plan.duration) {
                const U64 phase = time % plan.interpolation;
                const T* samples = window + time / plan.interpolation;
                const F32* taps = coefficients.data<F32>() + phase * plan.phaseLength;
                T value{};
                for (U64 tap = 0; tap < plan.phaseLength; ++tap) {
                    value += samples[tap] * taps[tap];
                }
                queued[count++] = value;
                time += plan.decimation;
            }
            time -= plan.duration;
            std::copy_n(window + plan.inputSamples, historySize, laneHistory);
        }

        if (count >= plan.outputPerLane) {
            for (U64 batch = 0; batch < plan.batchCount; ++batch) {
                T* destination = output.data<T>() + outputOffset + batch * outputBatchStride;
                const T* source = queued + batch * plan.outputSamples;
                for (U64 sample = 0; sample < plan.outputSamples; ++sample) {
                    destination[sample * outputSampleStride] = source[sample];
                }
            }
            // Before a submission fewer than Q samples remain; at most Q arrive.
            // A 2Q queue therefore bounds storage even for non-divisible buffers.
            const U64 remaining = count - plan.outputPerLane;
            std::memmove(queued, queued + plan.outputPerLane, remaining * sizeof(T));
        }
        finalTime = time;
        finalCount = count;
    }

    // Every lane shares the same rate and geometry, hence the same phase/count.
    nextTime = finalTime;
    if (finalCount < plan.outputPerLane) {
        pendingCount = finalCount;
        return Result::SKIP;
    }
    pendingCount = finalCount - plan.outputPerLane;
    return Result::SUCCESS;
}

JST_REGISTER_MODULE(RationalResamplerImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
