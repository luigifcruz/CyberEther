#include <jetstream/backend/devices/cuda/helpers.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>

#include <cstdint>
#include <limits>
#include <string_view>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

constexpr U64 kThreadsPerBlock = 256;
constexpr const char* kLayoutKernelName = "fft_layout";

static_assert(sizeof(CF32) == sizeof(cufftComplex));

constexpr const char* kLayoutKernelSource = R"(
struct alignas(8) KernelComplex {
    float real;
    float imag;
};

extern "C" __global__ void fft_layout(const unsigned char* input,
                                        unsigned char* output,
                                        unsigned long long elementCount,
                                        unsigned long long elementSize,
                                        unsigned long long offset,
                                        unsigned long long transformLength,
                                        unsigned long long spectrumLength,
                                        int rank,
                                        const unsigned long long* shape,
                                        const unsigned long long* stride,
                                        int mode,
                                        int conjugate) {
    const unsigned long long index =
        (static_cast<unsigned long long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (index >= elementCount) {
        return;
    }

    if (mode == 0) {
        unsigned long long remaining = index;
        unsigned long long sourceIndex = offset;
        for (int axis = rank - 1; axis >= 0; --axis) {
            const unsigned long long coordinate = remaining % shape[axis];
            remaining /= shape[axis];
            sourceIndex += coordinate * stride[axis];
        }

        const unsigned char* source = input + (sourceIndex * elementSize);
        unsigned char* destination = output + (index * elementSize);
        for (unsigned long long byte = 0; byte < elementSize; ++byte) {
            destination[byte] = source[byte];
        }
        return;
    }

    const unsigned long long batch = index / transformLength;
    const unsigned long long outputIndex = index % transformLength;
    const KernelComplex* spectrum = reinterpret_cast<const KernelComplex*>(input);

    if (mode == 1) {
        float value;
        if (outputIndex == 0) {
            value = spectrum[batch * spectrumLength].real;
        } else if ((outputIndex & 1ULL) != 0) {
            const unsigned long long frequency = (outputIndex + 1) / 2;
            value = spectrum[(batch * spectrumLength) + frequency].real;
        } else {
            const unsigned long long frequency = outputIndex / 2;
            value = spectrum[(batch * spectrumLength) + frequency].imag;
            if (conjugate != 0) {
                value = -value;
            }
        }

        reinterpret_cast<float*>(output)[index] = value;
        return;
    }

    KernelComplex value = {0.0f, 0.0f};
    if (outputIndex < spectrumLength) {
        value = spectrum[(batch * spectrumLength) + outputIndex];
        if (conjugate != 0) {
            value.imag = -value.imag;
        }
    }
    reinterpret_cast<KernelComplex*>(output)[index] = value;
}
)";

enum class TransformType {
    C2C,
    R2C,
    R2R,
};

enum class LayoutMode : I32 {
    Gather = 0,
    PackReal = 1,
    ExpandComplex = 2,
};

Result CheckCufft(const cufftResult status, const std::string_view operation) {
    if (status == CUFFT_SUCCESS) {
        return Result::SUCCESS;
    }

    JST_ERROR("[MODULE_FFT_NATIVE_CUDA] {} failed: {}",
              operation,
              cufftGetErrorString(status));
    return Result::ERROR;
}

}  // namespace

struct FftImplNativeCuda : public FftImpl,
                           public NativeCudaRuntimeContext,
                           public Scheduler::Context {
 public:
    Result create() final;

    Result computeInitialize() override;
    Result computeSubmit(const cudaStream_t& stream) override;
    Result computeDeinitialize() override;

 private:
    Result scheduleLayout(const cudaStream_t& stream,
                          const void* source,
                          void* destination,
                          LayoutMode mode);

    TransformType transformType = TransformType::C2C;
    cufftHandle plan = 0;
    bool planCreated = false;
    bool kernelCreated = false;
    bool requiresGather = false;

    U64 transformLength = 0;
    U64 spectrumLength = 0;
    U64 batchSize = 0;

    Tensor staging;
    Tensor spectrum;
    Tensor shapeTensor;
    Tensor strideTensor;
};

Result FftImplNativeCuda::create() {
    JST_CHECK(FftImpl::create());

    transformLength = input.shape(input.rank() - 1);
    spectrumLength = (transformLength / 2) + 1;
    batchSize = input.size() / transformLength;
    requiresGather = !input.contiguous();

    if (input.rank() > std::numeric_limits<I32>::max() ||
        transformLength > static_cast<U64>(std::numeric_limits<long long>::max()) ||
        batchSize > static_cast<U64>(std::numeric_limits<long long>::max())) {
        JST_ERROR("[MODULE_FFT_NATIVE_CUDA] Transform dimensions exceed cuFFT limits.");
        return Result::ERROR;
    }

    if (input.dtype() == DataType::CF32 && output.dtype() == DataType::CF32) {
        transformType = TransformType::C2C;
        return Result::SUCCESS;
    }

    if (input.dtype() == DataType::F32 && output.dtype() == DataType::CF32) {
        transformType = TransformType::R2C;
        return Result::SUCCESS;
    }

    if (input.dtype() == DataType::F32 && output.dtype() == DataType::F32) {
        transformType = TransformType::R2R;
        return Result::SUCCESS;
    }

    JST_ERROR("[MODULE_FFT_NATIVE_CUDA] Unsupported data type combination: {} -> {}.",
              input.dtype(), output.dtype());
    return Result::ERROR;
}

Result FftImplNativeCuda::computeInitialize() {
    JST_CHECK(CheckCufft(cufftCreate(&plan), "cufftCreate"));
    planCreated = true;

    long long dimensions[] = {static_cast<long long>(transformLength)};
    const auto cufftBatchSize = static_cast<long long>(batchSize);
    const cufftType cufftTransformType = transformType == TransformType::C2C
                                             ? CUFFT_C2C
                                             : CUFFT_R2C;
    std::size_t workSize = 0;

    JST_CHECK(CheckCufft(cufftMakePlanMany64(plan,
                                              1,
                                              dimensions,
                                              nullptr,
                                              1,
                                              0,
                                              nullptr,
                                              1,
                                              0,
                                              cufftTransformType,
                                              cufftBatchSize,
                                              &workSize),
                           "cufftMakePlanMany64"));

    if (transformType != TransformType::C2C) {
        JST_CHECK(spectrum.create(device(), DataType::CF32, {batchSize, spectrumLength}));
    }

    if (requiresGather) {
        if (transformType != TransformType::C2C) {
            JST_CHECK(staging.create(device(), DataType::F32, {batchSize, transformLength}));
        }

        Buffer::Config metadataConfig{};
        metadataConfig.hostAccessible = true;

        const Shape metadataShape = {static_cast<U64>(input.rank())};
        JST_CHECK(shapeTensor.create(device(), DataType::U64, metadataShape, metadataConfig));
        JST_CHECK(strideTensor.create(device(), DataType::U64, metadataShape, metadataConfig));

        auto* shapeData = shapeTensor.data<U64>();
        auto* strideData = strideTensor.data<U64>();
        for (Index axis = 0; axis < input.rank(); ++axis) {
            shapeData[axis] = input.shape(axis);
            strideData[axis] = input.stride(axis);
        }
    }

    if (requiresGather || transformType != TransformType::C2C) {
        JST_CHECK(createKernel(kLayoutKernelName, kLayoutKernelSource));
        kernelCreated = true;
    }

    return Result::SUCCESS;
}

Result FftImplNativeCuda::scheduleLayout(const cudaStream_t& stream,
                                         const void* source,
                                         void* destination,
                                         const LayoutMode mode) {
    void* sourceArgument = const_cast<void*>(source);
    U64 elementCount = input.size();
    U64 elementSize = input.elementSize();
    U64 offset = input.offset();
    I32 rank = static_cast<I32>(input.rank());
    void* shapeData = shapeTensor.data();
    void* strideData = strideTensor.data();
    I32 modeValue = static_cast<I32>(mode);
    I32 conjugate = forward ? 0 : 1;

    void* arguments[] = {
        &sourceArgument,
        &destination,
        &elementCount,
        &elementSize,
        &offset,
        &transformLength,
        &spectrumLength,
        &rank,
        &shapeData,
        &strideData,
        &modeValue,
        &conjugate,
    };

    const Extent3D<U64> block = {kThreadsPerBlock, 1, 1};
    const Extent3D<U64> grid = {
        (elementCount + kThreadsPerBlock - 1) / kThreadsPerBlock,
        1,
        1,
    };

    return scheduleKernel(kLayoutKernelName, stream, grid, block, arguments);
}

Result FftImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    JST_CHECK(CheckCufft(cufftSetStream(plan, stream), "cufftSetStream"));

    const auto* inputBase = static_cast<const std::uint8_t*>(input.buffer().data());
    void* outputData = output.buffer().data();
    if (!inputBase || !outputData) {
        JST_ERROR("[MODULE_FFT_NATIVE_CUDA] Missing input or output device buffer.");
        return Result::ERROR;
    }

    const void* contiguousInputData = inputBase + input.offsetBytes();

    if (transformType == TransformType::C2C) {
        void* transformInput = const_cast<void*>(contiguousInputData);
        if (requiresGather) {
            JST_CHECK(scheduleLayout(stream, inputBase, outputData, LayoutMode::Gather));
            transformInput = outputData;
        }

        const I32 direction = forward ? CUFFT_FORWARD : CUFFT_INVERSE;
        auto* transformInputData = reinterpret_cast<cufftComplex*>(transformInput);
        auto* transformOutputData = reinterpret_cast<cufftComplex*>(outputData);
        return CheckCufft(cufftExecC2C(plan,
                                       transformInputData,
                                       transformOutputData,
                                       direction),
                          "cufftExecC2C");
    }

    void* spectrumData = spectrum.buffer().data();
    if (!spectrumData) {
        JST_ERROR("[MODULE_FFT_NATIVE_CUDA] Missing cuFFT spectrum buffer.");
        return Result::ERROR;
    }

    void* transformInput = const_cast<void*>(contiguousInputData);
    if (requiresGather) {
        void* stagingData = staging.buffer().data();
        if (!stagingData) {
            JST_ERROR("[MODULE_FFT_NATIVE_CUDA] Missing cuFFT staging buffer.");
            return Result::ERROR;
        }

        JST_CHECK(scheduleLayout(stream, inputBase, stagingData, LayoutMode::Gather));
        transformInput = stagingData;
    }

    JST_CHECK(CheckCufft(cufftExecR2C(plan,
                                      reinterpret_cast<cufftReal*>(transformInput),
                                      reinterpret_cast<cufftComplex*>(spectrumData)),
                           "cufftExecR2C"));

    const LayoutMode outputMode = transformType == TransformType::R2R
                                      ? LayoutMode::PackReal
                                      : LayoutMode::ExpandComplex;
    return scheduleLayout(stream, spectrumData, outputData, outputMode);
}

Result FftImplNativeCuda::computeDeinitialize() {
    Result result = Result::SUCCESS;

    if (kernelCreated) {
        if (destroyKernel(kLayoutKernelName) != Result::SUCCESS) {
            result = Result::ERROR;
        }
        kernelCreated = false;
    }

    if (planCreated) {
        if (CheckCufft(cufftDestroy(plan), "cufftDestroy") != Result::SUCCESS) {
            result = Result::ERROR;
        }
        plan = 0;
        planCreated = false;
    }

    staging = {};
    spectrum = {};
    shapeTensor = {};
    strideTensor = {};

    return result;
}

JST_REGISTER_MODULE(FftImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
