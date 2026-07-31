#include <cmath>

#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

template<typename T>
void NormalizeLanes(const Tensor& input,
                    Tensor& output,
                    const Index sampleAxis,
                    const U64 laneCount) {
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

        F32 currentMax = 0.0f;
        for (U64 sample = 0; sample < sampleCount; ++sample) {
            currentMax = std::max(
                currentMax,
                static_cast<F32>(std::abs(
                    in[inputLaneOffset + sample * inputSampleStride])));
        }

        const F32 gain = currentMax != 0.0f ? 1.0f / currentMax : 1.0f;
        for (U64 sample = 0; sample < sampleCount; ++sample) {
            out[outputLaneOffset + sample * outputSampleStride] =
                in[inputLaneOffset + sample * inputSampleStride] * gain;
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
    NormalizeLanes<CF32>(input, output, sampleAxis, laneCount);
    return Result::SUCCESS;
}

Result AgcImplNativeCpu::kernelF32() {
    NormalizeLanes<F32>(input, output, sampleAxis, laneCount);
    return Result::SUCCESS;
}

JST_REGISTER_MODULE(AgcImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
