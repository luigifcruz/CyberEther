#include <algorithm>

#include <jetstream/backend/devices/cpu/helpers.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct FoldImplNativeCpu : public FoldImpl,
                           public NativeCpuRuntimeContext,
                           public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;

 private:
    Result kernelCF32();
    Result kernelF32();

    std::function<Result()> kernel;

    // Precomputed strides for coordinate conversion.
    std::vector<U64> inputStrides;
    std::vector<U64> outputStrides;
};

Result FoldImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(FoldImpl::create());

    // Precompute row-major strides for coordinate conversion.

    const auto& inShape = input.shape();
    const auto& outShape = output.shape();
    const U64 rank = input.rank();

    inputStrides.resize(rank);
    outputStrides.resize(rank);

    inputStrides[rank - 1] = 1;
    outputStrides[rank - 1] = 1;
    for (U64 d = rank - 1; d > 0; --d) {
        inputStrides[d - 1] = inputStrides[d] * inShape[d];
        outputStrides[d - 1] = outputStrides[d] * outShape[d];
    }

    // Register compute kernel.

    if (input.dtype() == DataType::CF32) {
        kernel = [this]() { return kernelCF32(); };
        return Result::SUCCESS;
    }

    if (input.dtype() == DataType::F32) {
        kernel = [this]() { return kernelF32(); };
        return Result::SUCCESS;
    }

    JST_ERROR("[MODULE_FOLD_NATIVE_CPU] Unsupported input "
              "data type: {}.", input.dtype());
    return Result::ERROR;
}

Result FoldImplNativeCpu::computeSubmit() {
    return kernel();
}

template<typename T>
static Result foldKernel(const Tensor& input,
                         Tensor& output,
                         const U64 foldAxis,
                         const U64 foldOffset,
                         const U64 foldSize,
                         const U64 decimFactor,
                         const std::vector<U64>& inStrides,
                         const std::vector<U64>& outStrides) {
    const U64 rank = input.rank();
    const U64 totalIn = input.size();
    const U64 totalOut = output.size();
    const auto& inShape = input.shape();

    T* outPtr = output.data<T>();
    const T* inPtr = input.data<T>();

    // Zero output buffer.
    std::fill(outPtr, outPtr + totalOut, T{});

    // Fold input into output.
    std::vector<U64> coords(rank);

    for (U64 i = 0; i < totalIn; ++i) {
        // Convert linear index to coordinates.
        U64 rem = i;
        for (U64 d = 0; d < rank; ++d) {
            coords[d] = rem / inStrides[d];
            rem %= inStrides[d];
        }

        // Apply offset and fold along axis.
        coords[foldAxis] = (coords[foldAxis] + foldOffset) % inShape[foldAxis];
        coords[foldAxis] %= foldSize;

        // Convert coordinates to output linear index.
        U64 outIdx = 0;
        for (U64 d = 0; d < rank; ++d) {
            outIdx += coords[d] * outStrides[d];
        }

        outPtr[outIdx] += inPtr[i];
    }

    // Average by decimation factor.
    const T divisor = static_cast<T>(decimFactor);
    for (U64 i = 0; i < totalOut; ++i) {
        outPtr[i] /= divisor;
    }

    return Result::SUCCESS;
}

Result FoldImplNativeCpu::kernelCF32() {
    return foldKernel<CF32>(input,
                            output,
                            axis,
                            offset,
                            size,
                            decimationFactor,
                            inputStrides,
                            outputStrides);
}

Result FoldImplNativeCpu::kernelF32() {
    return foldKernel<F32>(input,
                           output,
                           axis,
                           offset,
                           size,
                           decimationFactor,
                           inputStrides,
                           outputStrides);
}

JST_REGISTER_MODULE(FoldImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
