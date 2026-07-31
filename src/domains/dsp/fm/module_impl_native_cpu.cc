#include <cmath>
#include <complex>

#include <jetstream/backend/devices/cpu/helpers.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct FmImplNativeCpu : public FmImpl,
                         public NativeCpuRuntimeContext,
                         public Scheduler::Context {
 public:
    Result validate() override;
    Result computeSubmit() override;
};

Result FmImplNativeCpu::validate() {
    JST_CHECK(FmImpl::validate());

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_FM_NATIVE_CPU] Input must be complex (CF32).");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result FmImplNativeCpu::computeSubmit() {
    const CF32* inputData = input.data<CF32>();
    F32* outputData = output.data<F32>();
    const F32 refCoeff = ref;
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

        CF32 previous = previousSample[lane];
        bool hasPrevious = hasPreviousSample[lane] != 0;
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
                const CF32 current = inputData[inputOffset];
                outputData[outputOffset] = hasPrevious
                    ? std::arg(std::conj(previous) * current) * refCoeff
                    : 0.0f;
                previous = current;
                hasPrevious = true;
            }
        }

        previousSample[lane] = previous;
        hasPreviousSample[lane] = U8{1};
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(FmImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
