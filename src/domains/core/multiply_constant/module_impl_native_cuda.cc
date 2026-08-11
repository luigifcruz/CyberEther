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
constexpr const char* kMultiplyConstantKernelName = "multiply_constant_kernel";
static_assert(sizeof(CF32) == 2 * sizeof(F32));
static_assert(sizeof(CF64) == 2 * sizeof(F64));

constexpr const char* kMultiplyConstantKernelSource = R"(
<<<KERNEL_CONSTANTS>>>
extern "C" __global__ void multiply_constant_kernel(const unsigned char* input,
                                                      unsigned char* output,
                                                      float constant,
                                                      unsigned int dataType) {
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

    if (dataType == 0U) {
        reinterpret_cast<float*>(output)[index] =
            reinterpret_cast<const float*>(input)[inputIndex] * constant;
    } else if (dataType == 1U) {
        const float* complexInput =
            reinterpret_cast<const float*>(input) + (inputIndex * 2ULL);
        float* complexOutput = reinterpret_cast<float*>(output) + (index * 2ULL);
        complexOutput[0] = complexInput[0] * constant;
        complexOutput[1] = complexInput[1] * constant;
    } else if (dataType == 2U) {
        reinterpret_cast<double*>(output)[index] =
            reinterpret_cast<const double*>(input)[inputIndex] *
            static_cast<double>(constant);
    } else {
        const double* complexInput =
            reinterpret_cast<const double*>(input) + (inputIndex * 2ULL);
        double* complexOutput =
            reinterpret_cast<double*>(output) + (index * 2ULL);
        const double doubleConstant = static_cast<double>(constant);
        complexOutput[0] = complexInput[0] * doubleConstant;
        complexOutput[1] = complexInput[1] * doubleConstant;
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

struct MultiplyConstantImplNativeCuda : public MultiplyConstantImpl,
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

Result MultiplyConstantImplNativeCuda::validate() {
    JST_CHECK(MultiplyConstantImpl::validate());

    if (!inputs().contains("factor")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("factor").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() != DataType::F32 &&
        inputTensor.dtype() != DataType::CF32 &&
        inputTensor.dtype() != DataType::F64 &&
        inputTensor.dtype() != DataType::CF64) {
        JST_ERROR("[MODULE_MULTIPLY_CONSTANT_NATIVE_CUDA] Unsupported data type '{}'.",
                  inputTensor.dtype());
        return Result::ERROR;
    }

    const U64 blockCount = inputTensor.size() / kThreadsPerBlock +
                           (inputTensor.size() % kThreadsPerBlock != 0);
    if (blockCount > kMaxGridSizeX) {
        JST_ERROR("[MODULE_MULTIPLY_CONSTANT_NATIVE_CUDA] "
                  "Input size exceeds the CUDA grid limit.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result MultiplyConstantImplNativeCuda::create() {
    JST_CHECK(MultiplyConstantImpl::create());

    kernelPieces["KERNEL_CONSTANTS"] = BuildKernelConstants(input);
    return Result::SUCCESS;
}

Result MultiplyConstantImplNativeCuda::computeInitialize() {
    JST_CHECK(createKernel(kMultiplyConstantKernelName,
                           kMultiplyConstantKernelSource,
                           kernelPieces));
    kernelCreated = true;
    return Result::SUCCESS;
}

Result MultiplyConstantImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    const U64 elementCount = output.size();
    if (elementCount == 0) {
        return Result::SUCCESS;
    }

    const auto* inputBase = static_cast<const std::uint8_t*>(input.buffer().data());
    auto* outputBase = static_cast<std::uint8_t*>(output.buffer().data());
    if (!inputBase || !outputBase) {
        JST_ERROR("[MODULE_MULTIPLY_CONSTANT_NATIVE_CUDA] "
                  "Missing input or output device buffer.");
        return Result::ERROR;
    }

    U32 dataType = 0;
    switch (input.dtype()) {
        case DataType::F32:
            dataType = 0;
            break;
        case DataType::CF32:
            dataType = 1;
            break;
        case DataType::F64:
            dataType = 2;
            break;
        case DataType::CF64:
            dataType = 3;
            break;
        default:
            JST_ERROR("[MODULE_MULTIPLY_CONSTANT_NATIVE_CUDA] "
                      "Unsupported data type '{}'.",
                      input.dtype());
            return Result::ERROR;
    }

    const void* inputData = inputBase + input.offsetBytes();
    void* outputData = outputBase + output.offsetBytes();
    void* inputArgument = const_cast<void*>(inputData);
    F32 coefficient = constant;
    void* arguments[] = {&inputArgument, &outputData, &coefficient, &dataType};

    const Extent3D<U64> block = {kThreadsPerBlock, 1, 1};
    const Extent3D<U64> grid = {
        (elementCount + kThreadsPerBlock - 1) / kThreadsPerBlock,
        1,
        1,
    };
    return scheduleKernel(kMultiplyConstantKernelName, stream,
                          grid, block, arguments);
}

Result MultiplyConstantImplNativeCuda::computeDeinitialize() {
    if (kernelCreated) {
        JST_CHECK(destroyKernel(kMultiplyConstantKernelName));
    }
    kernelCreated = false;
    return Result::SUCCESS;
}

JST_REGISTER_MODULE(MultiplyConstantImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
