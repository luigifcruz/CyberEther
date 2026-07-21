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
constexpr const char* kRangeKernelName = "range_kernel";
constexpr const char* kRangeKernelSource = R"(
<<<KERNEL_CONSTANTS>>>
extern "C" __global__ void range_kernel(const float* input, float* output, float scale, float offset) {
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

    if (scale == 0.0f) {
        output[index] = 0.5f;
    } else {
        const float normalized = input[inputIndex] * scale + offset;
        output[index] = 0.5f + 0.5f * tanhf(4.0f * (normalized - 0.5f));
    }
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

}  // namespace

struct RangeImplNativeCuda : public RangeImpl,
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

Result RangeImplNativeCuda::create() {
    JST_CHECK(RangeImpl::create());

    if (input.dtype() != DataType::F32 || output.dtype() != DataType::F32) {
        JST_ERROR("[MODULE_RANGE_NATIVE_CUDA] Unsupported data type '{}'.", input.dtype());
        return Result::ERROR;
    }

    kernelPieces["KERNEL_CONSTANTS"] = BuildKernelConstants(input);
    return Result::SUCCESS;
}

Result RangeImplNativeCuda::computeInitialize() {
    JST_CHECK(createKernel(kRangeKernelName, kRangeKernelSource, kernelPieces));
    kernelCreated = true;
    return Result::SUCCESS;
}

Result RangeImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    const U64 elementCount = output.size();
    if (elementCount == 0) {
        return Result::SUCCESS;
    }

    const auto* inputBase = static_cast<const std::uint8_t*>(input.buffer().data());
    auto* outputBase = static_cast<std::uint8_t*>(output.buffer().data());
    if (!inputBase || !outputBase) {
        JST_ERROR("[MODULE_RANGE_NATIVE_CUDA] Missing input or output device buffer.");
        return Result::ERROR;
    }

    const void* inputData = inputBase + input.offsetBytes();
    void* outputData = outputBase + output.offsetBytes();
    void* inputArgument = const_cast<void*>(inputData);
    F32 scale = scalingCoeff;
    F32 offset = offsetCoeff;
    void* arguments[] = {&inputArgument, &outputData, &scale, &offset};

    const Extent3D<U64> block = {kThreadsPerBlock, 1, 1};
    const Extent3D<U64> grid = {
        (elementCount + kThreadsPerBlock - 1) / kThreadsPerBlock,
        1,
        1,
    };
    return scheduleKernel(kRangeKernelName, stream, grid, block, arguments);
}

Result RangeImplNativeCuda::computeDeinitialize() {
    if (kernelCreated) {
        JST_CHECK(destroyKernel(kRangeKernelName));
    }
    kernelCreated = false;
    return Result::SUCCESS;
}

JST_REGISTER_MODULE(RangeImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
