#include <algorithm>
#include <array>
#include <limits>

#include <jetstream/backend/devices/cpu/helpers.hh>
#include <jetstream/memory/macros.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct RrcFilterImplNativeCpu : public RrcFilterImpl,
                                public NativeCpuRuntimeContext,
                                public Scheduler::Context {
 public:
    Result validate() final;
    Result create() final;

    Result computeSubmit() override;

 private:
    template<typename T>
    Result filterKernel();

    Tensor workspace;
    std::function<Result()> kernel;
};

Result RrcFilterImplNativeCpu::validate() {
    JST_CHECK(RrcFilterImpl::validate());

    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("buffer").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() != DataType::CF32 &&
        inputTensor.dtype() != DataType::F32) {
        JST_ERROR("[MODULE_RRC_FILTER_NATIVE_CPU] Unsupported input data type: {}.",
                  inputTensor.dtype());
        return Result::ERROR;
    }

    U64 coefficientBytes = 0;
    U64 historyElements = 0;
    U64 historyBytes = 0;
    U64 workspaceElements = 0;
    U64 workspaceBytes = 0;
    U64 alignedBytes = 0;
    if (!detail::CheckedMultiply(candidate()->taps,
                                 static_cast<U64>(sizeof(F32)),
                                 coefficientBytes) ||
        !detail::CheckedPageAlignedSize(coefficientBytes, alignedBytes) ||
        alignedBytes > std::numeric_limits<std::size_t>::max() ||
        !detail::CheckedMultiply(candidate()->taps - 1,
                                 validatedLaneCount,
                                 historyElements) ||
        !detail::CheckedMultiply(historyElements,
                                 static_cast<U64>(DataTypeSize(inputTensor.dtype())),
                                 historyBytes) ||
        !detail::CheckedPageAlignedSize(historyBytes, alignedBytes) ||
        alignedBytes > std::numeric_limits<std::size_t>::max() ||
        !detail::CheckedAdd(candidate()->taps - 1,
                            inputTensor.shape(*validatedSignalAxes.sample),
                            workspaceElements) ||
        !detail::CheckedMultiply(workspaceElements,
                                 static_cast<U64>(DataTypeSize(inputTensor.dtype())),
                                 workspaceBytes) ||
        !detail::CheckedPageAlignedSize(workspaceBytes, alignedBytes) ||
        alignedBytes > std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_RRC_FILTER_NATIVE_CPU] Filter buffers exceed the supported allocation range.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result RrcFilterImplNativeCpu::create() {
    JST_CHECK(RrcFilterImpl::create());
    JST_CHECK(workspace.create(input.device(), input.dtype(),
                               {taps - 1 + input.shape(*signalAxes.sample)}));

    if (input.dtype() == DataType::CF32) {
        kernel = [this]() { return filterKernel<CF32>(); };
    } else {
        kernel = [this]() { return filterKernel<F32>(); };
    }

    return Result::SUCCESS;
}

Result RrcFilterImplNativeCpu::computeSubmit() {
    return kernel();
}

namespace {

template<typename T>
void convolve(const T* samples,
              T* output,
              const F32* coefficients,
              const U64 sampleCount,
              const U64 taps,
              const U64 outputStride) {
    constexpr U64 blockSize = 32;
    constexpr U64 componentCount = blockSize * sizeof(T) / sizeof(F32);
    const U64 blockEnd = outputStride == 1
        ? sampleCount - sampleCount % blockSize : 0;

    for (U64 sample = 0; sample < blockEnd; sample += blockSize) {
        std::array<F32, componentCount> sums{};
        for (U64 tap = 0; tap < taps; ++tap) {
            const F32* window = reinterpret_cast<const F32*>(
                samples + sample + taps - 1 - tap);
            for (U64 component = 0; component < componentCount; ++component) {
                sums[component] += window[component] * coefficients[tap];
            }
        }
        std::copy(sums.begin(), sums.end(), reinterpret_cast<F32*>(output + sample));
    }

    for (U64 sample = blockEnd; sample < sampleCount; ++sample) {
        T sum{};
        for (U64 tap = 0; tap < taps; ++tap) {
            sum += samples[sample + taps - 1 - tap] * coefficients[tap];
        }
        output[sample * outputStride] = sum;
    }
}

}

template<typename T>
Result RrcFilterImplNativeCpu::filterKernel() {
    const T* inPtr = input.data<T>();
    T* outPtr = output.data<T>();
    T* histPtr = history.data<T>();
    T* window = workspace.data<T>();
    const F32* coeffPtr = coeffs.data<F32>();
    const U64 historyCount = taps - 1;
    const Index sampleAxis = *signalAxes.sample;
    const U64 sampleCount = input.shape(sampleAxis);
    const U64 inputSampleStride = input.stride(sampleAxis);
    const U64 outputSampleStride = output.stride(sampleAxis);
    const U64 batchCount = signalAxes.batch
        ? input.shape(*signalAxes.batch) : 1;
    const U64 inputBatchStride = signalAxes.batch
        ? input.stride(*signalAxes.batch) : 0;
    const U64 outputBatchStride = signalAxes.batch
        ? output.stride(*signalAxes.batch) : 0;

    for (U64 lane = 0; lane < laneCount; ++lane) {
        U64 coordinates = lane;
        U64 inputLaneOffset = 0;
        U64 outputLaneOffset = 0;
        for (Index axis = input.rank(); axis-- > 0;) {
            if (axis == sampleAxis ||
                (signalAxes.batch && axis == *signalAxes.batch)) {
                continue;
            }
            const U64 coordinate = coordinates % input.shape(axis);
            coordinates /= input.shape(axis);
            inputLaneOffset += coordinate * input.stride(axis);
            outputLaneOffset += coordinate * output.stride(axis);
        }

        T* laneHistory = histPtr + lane * historyCount;
        for (U64 batch = 0; batch < batchCount; ++batch) {
            const T* batchInput = inPtr + inputLaneOffset + batch * inputBatchStride;
            T* batchOutput = outPtr + outputLaneOffset + batch * outputBatchStride;

            std::copy_n(laneHistory, historyCount, window);
            if (inputSampleStride == 1) {
                std::copy_n(batchInput, sampleCount, window + historyCount);
            } else {
                for (U64 sample = 0; sample < sampleCount; ++sample) {
                    window[historyCount + sample] = batchInput[sample * inputSampleStride];
                }
            }

            convolve(window, batchOutput, coeffPtr, sampleCount, taps, outputSampleStride);

            std::copy_n(window + sampleCount, historyCount, laneHistory);
        }
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(RrcFilterImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
