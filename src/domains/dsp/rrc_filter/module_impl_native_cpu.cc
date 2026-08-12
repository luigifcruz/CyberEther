#include <cstring>
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
    Result kernelCF32();
    Result kernelF32();

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
    U64 alignedBytes = 0;
    if (!detail::CheckedMultiply(candidate()->taps,
                                 static_cast<U64>(sizeof(F32)),
                                 coefficientBytes) ||
        !detail::CheckedPageAlignedSize(coefficientBytes, alignedBytes) ||
        alignedBytes > std::numeric_limits<std::size_t>::max() ||
        !detail::CheckedMultiply(candidate()->taps,
                                 validatedLaneCount,
                                 historyElements) ||
        !detail::CheckedMultiply(historyElements,
                                 static_cast<U64>(DataTypeSize(inputTensor.dtype())),
                                 historyBytes) ||
        !detail::CheckedPageAlignedSize(historyBytes, alignedBytes) ||
        alignedBytes > std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_RRC_FILTER_NATIVE_CPU] Tap buffers exceed the supported allocation range.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result RrcFilterImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(RrcFilterImpl::create());

    // Register compute kernel.

    if (input.dtype() == DataType::CF32) {
        kernel = [this]() { return kernelCF32(); };
    } else {
        kernel = [this]() { return kernelF32(); };
    }

    return Result::SUCCESS;
}

Result RrcFilterImplNativeCpu::computeSubmit() {
    return kernel();
}

template<typename T>
static Result filterKernel(const Tensor& input,
                           Tensor& output,
                           const Tensor& coeffs,
                           Tensor& history,
                           std::vector<U64>& historyIndex,
                           const SignalAxes& signalAxes,
                           const U64 laneCount,
                           const U64 taps) {
    const U64 numTaps = taps;
    const T* inPtr = input.data<T>();
    T* outPtr = output.data<T>();
    T* histPtr = history.data<T>();
    const F32* coeffPtr = coeffs.data<F32>();
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

        T* laneHistory = histPtr + lane * numTaps;
        U64 index = historyIndex[lane];
        for (U64 batch = 0; batch < batchCount; ++batch) {
            const U64 inputBatchOffset = inputLaneOffset +
                                         batch * inputBatchStride;
            const U64 outputBatchOffset = outputLaneOffset +
                                          batch * outputBatchStride;
            for (U64 sample = 0; sample < sampleCount; ++sample) {
                const U64 inputOffset = inputBatchOffset +
                                        sample * inputSampleStride;
                const U64 outputOffset = outputBatchOffset +
                                         sample * outputSampleStride;
                laneHistory[index] = inPtr[inputOffset];

                T outputSample{};
                for (U64 k = 0; k < numTaps; ++k) {
                    const U64 historyOffset =
                        (index + numTaps - k) % numTaps;
                    outputSample += laneHistory[historyOffset] * coeffPtr[k];
                }
                outPtr[outputOffset] = outputSample;
                index = (index + 1) % numTaps;
            }
        }
        historyIndex[lane] = index;
    }

    return Result::SUCCESS;
}

Result RrcFilterImplNativeCpu::kernelCF32() {
    return filterKernel<CF32>(input,
                              output,
                              coeffs,
                              history,
                              historyIndex,
                              signalAxes,
                              laneCount,
                              taps);
}

Result RrcFilterImplNativeCpu::kernelF32() {
    return filterKernel<F32>(input,
                             output,
                             coeffs,
                             history,
                             historyIndex,
                             signalAxes,
                             laneCount,
                             taps);
}

JST_REGISTER_MODULE(RrcFilterImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
