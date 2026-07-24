#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

constexpr U64 kThreadsPerBlock = 256;
constexpr U64 kMaxGridSizeX = std::numeric_limits<I32>::max();
constexpr const char* kMultiplyKernelName = "multiply_kernel";
constexpr const char* kMultiplyKernelSource = R"(
struct alignas(8) KernelComplex {
    float real;
    float imag;
};

<<<KERNEL_CONSTANTS>>>
<<<VALUE_DECLS>>>
extern "C" __global__ void multiply_kernel(const Value* a, const Value* b, Value* c) {
    const unsigned long long index =
        (static_cast<unsigned long long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (index >= kElementCount) {
        return;
    }

    unsigned long long remaining = index;
    unsigned long long indexA = 0;
    unsigned long long indexB = 0;
    for (int axis = kRank - 1; axis >= 0; --axis) {
        const unsigned long long coordinate = remaining % kShape[axis];
        remaining /= kShape[axis];
        indexA += coordinate * kStrideA[axis];
        indexB += coordinate * kStrideB[axis];
    }

    c[index] = MultiplyValue(a[indexA], b[indexB]);
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

std::string BuildKernelConstants(const Tensor& a, const Tensor& b, const Tensor& c) {
    return jst::fmt::format(
        "static constexpr unsigned long long kElementCount = {}ULL;\n"
        "static constexpr int kRank = {};\n"
        "static constexpr unsigned long long kShape[] = {};\n"
        "static constexpr unsigned long long kStrideA[] = {};\n"
        "static constexpr unsigned long long kStrideB[] = {};\n",
        c.size(),
        c.rank(),
        MakeU64ArrayLiteral(c.shape()),
        MakeU64ArrayLiteral(a.stride()),
        MakeU64ArrayLiteral(b.stride())
    );
}

const char* BuildValueDecls(const DataType dtype) {
    if (dtype == DataType::F32) {
        return R"(
using Value = float;
__device__ __forceinline__ Value MultiplyValue(const Value a, const Value b) {
    return a * b;
}
)";
    }

    return R"(
using Value = KernelComplex;
__device__ __forceinline__ Value MultiplyValue(const Value a, const Value b) {
    Value result;
    result.real = (a.real * b.real) - (a.imag * b.imag);
    result.imag = (a.real * b.imag) + (a.imag * b.real);
    return result;
}
)";
}

}  // namespace

struct MultiplyImplNativeCuda : public MultiplyImpl,
                                public NativeCudaRuntimeContext,
                                public Scheduler::Context {
 public:
    Result validate() final;
    Result create() final;

    Result computeInitialize() override;
    Result computeSubmit(const cudaStream_t& stream) override;
    Result computeDeinitialize() override;

 private:
    bool kernelCreated = false;
    std::unordered_map<std::string, std::string> kernelPieces;
};

Result MultiplyImplNativeCuda::validate() {
    JST_CHECK(MultiplyImpl::validate());

    if (!inputs().contains("a") || !inputs().contains("b")) {
        return Result::SUCCESS;
    }

    const Tensor& tensorA = inputs().at("a").tensor;
    const Tensor& tensorB = inputs().at("b").tensor;
    if (!tensorA.validShape() || !tensorB.validShape() ||
        tensorA.size() == 0 || tensorB.size() == 0) {
        return Result::SUCCESS;
    }

    if (tensorA.dtype() != tensorB.dtype()) {
        JST_ERROR("[MODULE_MULTIPLY_NATIVE_CUDA] Input data types '{}' and '{}' do not match.",
                  tensorA.dtype(), tensorB.dtype());
        return Result::ERROR;
    }

    if (tensorA.dtype() != DataType::F32 && tensorA.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_MULTIPLY_NATIVE_CUDA] Unsupported data type '{}'.",
                  tensorA.dtype());
        return Result::ERROR;
    }

    const U64 blockCount = validatedOutputElementCount / kThreadsPerBlock +
                           (validatedOutputElementCount % kThreadsPerBlock != 0);
    if (blockCount > kMaxGridSizeX) {
        JST_ERROR("[MODULE_MULTIPLY_NATIVE_CUDA] Output size exceeds the CUDA grid limit.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result MultiplyImplNativeCuda::create() {
    JST_CHECK(MultiplyImpl::create());

    kernelPieces["KERNEL_CONSTANTS"] = BuildKernelConstants(a, b, c);
    kernelPieces["VALUE_DECLS"] = BuildValueDecls(a.dtype());
    return Result::SUCCESS;
}

Result MultiplyImplNativeCuda::computeInitialize() {
    JST_CHECK(createKernel(kMultiplyKernelName, kMultiplyKernelSource, kernelPieces));
    kernelCreated = true;
    return Result::SUCCESS;
}

Result MultiplyImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    const U64 elementCount = c.size();
    if (elementCount == 0) {
        return Result::SUCCESS;
    }

    const auto* aBase = static_cast<const std::uint8_t*>(a.buffer().data());
    const auto* bBase = static_cast<const std::uint8_t*>(b.buffer().data());
    auto* cBase = static_cast<std::uint8_t*>(c.buffer().data());
    if (!aBase || !bBase || !cBase) {
        JST_ERROR("[MODULE_MULTIPLY_NATIVE_CUDA] Missing input or output device buffer.");
        return Result::ERROR;
    }

    const void* aData = aBase + a.offsetBytes();
    const void* bData = bBase + b.offsetBytes();
    void* cData = cBase + c.offsetBytes();
    void* aArgument = const_cast<void*>(aData);
    void* bArgument = const_cast<void*>(bData);
    void* arguments[] = {&aArgument, &bArgument, &cData};

    const Extent3D<U64> block = {kThreadsPerBlock, 1, 1};
    const Extent3D<U64> grid = {
        elementCount / kThreadsPerBlock + (elementCount % kThreadsPerBlock != 0),
        1,
        1,
    };
    return scheduleKernel(kMultiplyKernelName, stream, grid, block, arguments);
}

Result MultiplyImplNativeCuda::computeDeinitialize() {
    if (kernelCreated) {
        JST_CHECK(destroyKernel(kMultiplyKernelName));
    }
    kernelCreated = false;
    return Result::SUCCESS;
}

JST_REGISTER_MODULE(MultiplyImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
