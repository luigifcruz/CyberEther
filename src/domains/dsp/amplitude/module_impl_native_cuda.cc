#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

constexpr U64 kThreadsPerBlock = 256;
constexpr const char* kAmplitudeKernelName = "amplitude_kernel";
constexpr const char* kAmplitudeKernelSource = R"(
struct alignas(8) KernelComplex {
    float real;
    float imag;
};

<<<KERNEL_CONSTANTS>>>
<<<INPUT_DECLS>>>
__device__ __forceinline__ float ApproxLog10(const float value) {
    int exponent;
    const float fraction = frexpf(fabsf(value), &exponent);
    float result = 1.23149591368684f;
    result *= fraction;
    result += -4.11852516267426f;
    result *= fraction;
    result += 6.02197014179219f;
    result *= fraction;
    result += -3.13396450166353f;
    result += exponent;
    return result * 0.3010299956639812f;
}

extern "C" __global__ void amplitude_kernel(const InputValue* input, float* output, float scalingCoeff) {
    const unsigned long long index =
        (static_cast<unsigned long long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (index >= kElementCount) {
        return;
    }

    unsigned long long remaining = index;
    unsigned long long inputIndex = 0;
    for (int axis = kRank - 1; axis >= 0; --axis) {
        const unsigned long long coordinate = remaining % kShape[axis];
        remaining /= kShape[axis];
        inputIndex += coordinate * kInputStride[axis];
    }

    const float magnitude = Magnitude(input[inputIndex]);
    if (magnitude == 0.0f) {
        output[index] = -__int_as_float(0x7f800000);
        return;
    }
    output[index] = 20.0f * ApproxLog10(magnitude) + scalingCoeff;
}
)";

std::string MakeU64ArrayLiteral(const Shape& values) {
    if (values.empty()) {
        return "{0ULL}";
    }

    std::vector<std::string> formattedValues;
    formattedValues.reserve(values.size());
    for (const auto value : values) {
        formattedValues.push_back(jst::fmt::format("{}ULL", value));
    }
    return jst::fmt::format("{{{}}}", jst::fmt::join(formattedValues, ", "));
}

std::string BuildKernelConstants(const Tensor& input) {
    return jst::fmt::format(
        "static constexpr unsigned long long kElementCount = {}ULL;\n"
        "static constexpr int kRank = {};\n"
        "static constexpr unsigned long long kShape[] = {};\n"
        "static constexpr unsigned long long kInputStride[] = {};\n",
        input.size(),
        input.rank(),
        MakeU64ArrayLiteral(input.shape()),
        MakeU64ArrayLiteral(input.stride())
    );
}

const char* BuildInputDecls(const DataType dtype) {
    if (dtype == DataType::F32) {
        return R"(
using InputValue = float;
__device__ __forceinline__ float Magnitude(const InputValue value) {
    return fabsf(value);
}
)";
    }

    if (dtype == DataType::CF32) {
        return R"(
using InputValue = KernelComplex;
__device__ __forceinline__ float Magnitude(const InputValue value) {
    return sqrtf((value.real * value.real) + (value.imag * value.imag));
}
)";
    }

    return nullptr;
}

}  // namespace

struct AmplitudeImplNativeCuda : public AmplitudeImpl,
                                 public NativeCudaRuntimeContext,
                                 public Scheduler::Context {
 public:
    Result create() final;

    Result computeInitialize() override;
    Result computeSubmit(const cudaStream_t& stream) override;
    Result computeDeinitialize() override;

 private:
    bool kernelCreated = false;
    std::unordered_map<std::string, std::string> kernelPieces;
};

Result AmplitudeImplNativeCuda::create() {
    JST_CHECK(AmplitudeImpl::create());

    const char* inputDecls = BuildInputDecls(input.dtype());
    if (!inputDecls) {
        JST_ERROR("[MODULE_AMPLITUDE_NATIVE_CUDA] Unsupported input data type: {}.", input.dtype());
        return Result::ERROR;
    }

    kernelPieces["KERNEL_CONSTANTS"] = BuildKernelConstants(input);
    kernelPieces["INPUT_DECLS"] = inputDecls;
    return Result::SUCCESS;
}

Result AmplitudeImplNativeCuda::computeInitialize() {
    JST_CHECK(createKernel(kAmplitudeKernelName, kAmplitudeKernelSource, kernelPieces));
    kernelCreated = true;
    return Result::SUCCESS;
}

Result AmplitudeImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    const U64 elementCount = output.size();
    if (elementCount == 0) {
        return Result::SUCCESS;
    }

    const auto* inputBase = static_cast<const std::uint8_t*>(input.buffer().data());
    auto* outputBase = static_cast<std::uint8_t*>(output.buffer().data());
    if (!inputBase || !outputBase) {
        JST_ERROR("[MODULE_AMPLITUDE_NATIVE_CUDA] Missing input or output device buffer.");
        return Result::ERROR;
    }

    const void* inputData = inputBase + input.offsetBytes();
    void* outputData = outputBase + output.offsetBytes();
    void* inputArgument = const_cast<void*>(inputData);
    F32 coefficient = scalingCoeff;
    void* arguments[] = {&inputArgument, &outputData, &coefficient};

    const Extent3D<U64> block = {kThreadsPerBlock, 1, 1};
    const Extent3D<U64> grid = {
        (elementCount + kThreadsPerBlock - 1) / kThreadsPerBlock,
        1,
        1,
    };
    return scheduleKernel(kAmplitudeKernelName, stream, grid, block, arguments);
}

Result AmplitudeImplNativeCuda::computeDeinitialize() {
    if (kernelCreated) {
        JST_CHECK(destroyKernel(kAmplitudeKernelName));
    }
    kernelCreated = false;
    return Result::SUCCESS;
}

JST_REGISTER_MODULE(AmplitudeImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
