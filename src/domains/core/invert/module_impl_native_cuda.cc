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
constexpr const char* kInvertKernelName = "invert_kernel";
constexpr const char* kInvertKernelSource = R"(
struct alignas(8) KernelComplex {
    float real;
    float imag;
};

<<<KERNEL_CONSTANTS>>>
extern "C" __global__ void invert_kernel(const KernelComplex* input, KernelComplex* output) {
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

    KernelComplex value = input[inputIndex];
    const unsigned long long axisCoordinate =
        (index / kAxisInnerSize) % kAxisLength;
    if ((axisCoordinate & 1ULL) != 0) {
        value.real = -value.real;
        value.imag = -value.imag;
    }
    output[index] = value;
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

std::string BuildKernelConstants(const Tensor& input, const Index axis) {
    U64 axisInnerSize = 1;
    for (Index axisIndex = axis + 1; axisIndex < input.rank(); ++axisIndex) {
        axisInnerSize *= input.shape(axisIndex);
    }
    if (axisInnerSize == 0) {
        axisInnerSize = 1;
    }
    const U64 axisLength = input.shape(axis) == 0 ? 1 : input.shape(axis);

    return jst::fmt::format(
        "static constexpr unsigned long long kElementCount = {}ULL;\n"
        "static constexpr int kRank = {};\n"
        "static constexpr unsigned long long kAxisLength = {}ULL;\n"
        "static constexpr unsigned long long kAxisInnerSize = {}ULL;\n"
        "static constexpr unsigned long long kShape[] = {};\n"
        "static constexpr unsigned long long kInputStride[] = {};\n",
        input.size(),
        input.rank(),
        axisLength,
        axisInnerSize,
        MakeU64ArrayLiteral(input.shape()),
        MakeU64ArrayLiteral(input.stride())
    );
}

}  // namespace

struct InvertImplNativeCuda : public InvertImpl,
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

Result InvertImplNativeCuda::create() {
    JST_CHECK(InvertImpl::create());

    if (input.dtype() != DataType::CF32 || output.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_INVERT_NATIVE_CUDA] Unsupported data type '{}'.", input.dtype());
        return Result::ERROR;
    }

    kernelPieces["KERNEL_CONSTANTS"] = BuildKernelConstants(input, resolvedAxis);
    return Result::SUCCESS;
}

Result InvertImplNativeCuda::computeInitialize() {
    JST_CHECK(createKernel(kInvertKernelName, kInvertKernelSource, kernelPieces));
    kernelCreated = true;
    return Result::SUCCESS;
}

Result InvertImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    const U64 elementCount = output.size();
    if (elementCount == 0) {
        return Result::SUCCESS;
    }

    const auto* inputBase = static_cast<const std::uint8_t*>(input.buffer().data());
    auto* outputBase = static_cast<std::uint8_t*>(output.buffer().data());
    if (!inputBase || !outputBase) {
        JST_ERROR("[MODULE_INVERT_NATIVE_CUDA] Missing input or output device buffer.");
        return Result::ERROR;
    }

    const void* inputData = inputBase + input.offsetBytes();
    void* outputData = outputBase + output.offsetBytes();
    void* inputArgument = const_cast<void*>(inputData);
    void* arguments[] = {&inputArgument, &outputData};

    const Extent3D<U64> block = {kThreadsPerBlock, 1, 1};
    const Extent3D<U64> grid = {
        (elementCount + kThreadsPerBlock - 1) / kThreadsPerBlock,
        1,
        1,
    };
    return scheduleKernel(kInvertKernelName, stream, grid, block, arguments);
}

Result InvertImplNativeCuda::computeDeinitialize() {
    if (kernelCreated) {
        JST_CHECK(destroyKernel(kInvertKernelName));
    }
    kernelCreated = false;
    return Result::SUCCESS;
}

JST_REGISTER_MODULE(InvertImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
