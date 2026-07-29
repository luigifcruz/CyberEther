#include <algorithm>
#include <complex>
#include <limits>
#include <type_traits>

#include <jetstream/backend/devices/cpu/helpers.hh>
#include <jetstream/memory/macros.hh>
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
    Result validate() final;
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

Result FoldImplNativeCpu::validate() {
    JST_CHECK(FoldImpl::validate());

    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("buffer").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() != DataType::CF32 &&
        inputTensor.dtype() != DataType::F32) {
        JST_ERROR("[MODULE_FOLD_NATIVE_CPU] Unsupported input data type: {}.",
                  inputTensor.dtype());
        return Result::ERROR;
    }

    U64 alignedOutputSize = 0;
    if (!detail::CheckedPageAlignedSize(validatedOutputSizeBytes, alignedOutputSize) ||
        alignedOutputSize > std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_FOLD_NATIVE_CPU] Output allocation size is too large.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

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
    } else {
        kernel = [this]() { return kernelF32(); };
    }

    return Result::SUCCESS;
}

Result FoldImplNativeCpu::computeSubmit() {
    return kernel();
}

template<typename T>
static Result foldKernel(const Tensor& input,
                         Tensor& output,
                         const Index foldAxis,
                         const U64 foldOffset,
                         const U64 foldSize,
                         const U64 decimFactor,
                         const std::vector<U64>& inStrides,
                         const std::vector<U64>& outStrides) {
    const U64 rank = input.rank();
    const U64 totalOut = output.size();
    const auto& inShape = input.shape();

    T* outPtr = output.data<T>();
    const T* inPtr = input.data<T>();
    const F64 divisor = static_cast<F64>(decimFactor);
    const U64 axisSize = inShape[foldAxis];
    const U64 normalizedOffset = foldOffset % axisSize;
    std::vector<U64> coords(rank);

    for (U64 outputIndex = 0; outputIndex < totalOut; ++outputIndex) {
        U64 rem = outputIndex;
        for (U64 d = 0; d < rank; ++d) {
            coords[d] = rem / outStrides[d];
            rem %= outStrides[d];
        }

        U64 inputBaseIndex = 0;
        for (U64 d = 0; d < rank; ++d) {
            if (d != foldAxis) {
                inputBaseIndex += coords[d] * inStrides[d];
            }
        }

        if constexpr (std::is_same_v<T, F32>) {
            F64 sum = 0.0;
            for (U64 group = 0; group < decimFactor; ++group) {
                const U64 shiftedAxis = coords[foldAxis] + group * foldSize;
                const U64 inputAxis = shiftedAxis >= normalizedOffset
                    ? shiftedAxis - normalizedOffset
                    : axisSize - (normalizedOffset - shiftedAxis);
                const U64 inputIndex = inputBaseIndex +
                                       inputAxis * inStrides[foldAxis];
                sum += static_cast<F64>(inPtr[inputIndex]);
            }
            outPtr[outputIndex] = static_cast<F32>(sum / divisor);
        } else {
            std::complex<F64> sum{0.0, 0.0};
            for (U64 group = 0; group < decimFactor; ++group) {
                const U64 shiftedAxis = coords[foldAxis] + group * foldSize;
                const U64 inputAxis = shiftedAxis >= normalizedOffset
                    ? shiftedAxis - normalizedOffset
                    : axisSize - (normalizedOffset - shiftedAxis);
                const U64 inputIndex = inputBaseIndex +
                                       inputAxis * inStrides[foldAxis];
                sum += std::complex<F64>{inPtr[inputIndex].real(),
                                         inPtr[inputIndex].imag()};
            }
            sum /= divisor;
            outPtr[outputIndex] = T{static_cast<F32>(sum.real()),
                                    static_cast<F32>(sum.imag())};
        }
    }

    return Result::SUCCESS;
}

Result FoldImplNativeCpu::kernelCF32() {
    return foldKernel<CF32>(input,
                            output,
                            resolvedAxis,
                            offset,
                            size,
                            decimationFactor,
                            inputStrides,
                            outputStrides);
}

Result FoldImplNativeCpu::kernelF32() {
    return foldKernel<F32>(input,
                           output,
                           resolvedAxis,
                           offset,
                           size,
                           decimationFactor,
                           inputStrides,
                           outputStrides);
}

JST_REGISTER_MODULE(FoldImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
