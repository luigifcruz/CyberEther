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
constexpr const char* kCastKernelName = "cast_kernel";

constexpr const char* kCastContiguousKernelSource = R"(
template<typename T>
struct KernelComplex {
    T real;
    T imag;
};

<<<KERNEL_CONSTANTS>>>
<<<CONVERSION_DECLS>>>
extern "C" __global__ void cast_kernel(const InputValue* input,
                                        OutputValue* output) {
    const unsigned long long index =
        (static_cast<unsigned long long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (index >= kElementCount) {
        return;
    }

    output[index] = ConvertValue(input[index]);
}
)";

constexpr const char* kCastStridedKernelSource = R"(
template<typename T>
struct KernelComplex {
    T real;
    T imag;
};

<<<KERNEL_CONSTANTS>>>
<<<CONVERSION_DECLS>>>
extern "C" __global__ void cast_kernel(const InputValue* input,
                                        OutputValue* output) {
    const unsigned long long index =
        (static_cast<unsigned long long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (index >= kElementCount) {
        return;
    }

    unsigned long long remaining = index;
    unsigned long long sourceIndex = 0;
    for (int axis = kRank - 1; axis >= 0; --axis) {
        const unsigned long long coordinate = remaining % kShape[axis];
        remaining /= kShape[axis];
        sourceIndex += coordinate * kStride[axis];
    }

    output[index] = ConvertValue(input[sourceIndex]);
}
)";

template<typename T>
struct KernelComplexHost {
    T real;
    T imag;
};

static_assert(sizeof(CF32) == sizeof(KernelComplexHost<F32>));
static_assert(alignof(CF32) == alignof(KernelComplexHost<F32>));
static_assert(sizeof(CI8) == sizeof(KernelComplexHost<I8>));
static_assert(alignof(CI8) == alignof(KernelComplexHost<I8>));
static_assert(sizeof(CI16) == sizeof(KernelComplexHost<I16>));
static_assert(alignof(CI16) == alignof(KernelComplexHost<I16>));
static_assert(sizeof(CI32) == sizeof(KernelComplexHost<I32>));
static_assert(alignof(CI32) == alignof(KernelComplexHost<I32>));
static_assert(sizeof(CU8) == sizeof(KernelComplexHost<U8>));
static_assert(alignof(CU8) == alignof(KernelComplexHost<U8>));
static_assert(sizeof(CU16) == sizeof(KernelComplexHost<U16>));
static_assert(alignof(CU16) == alignof(KernelComplexHost<U16>));
static_assert(sizeof(CU32) == sizeof(KernelComplexHost<U32>));
static_assert(alignof(CU32) == alignof(KernelComplexHost<U32>));

std::string MakeU64Literal(const U64 value) {
    return jst::fmt::format("{}ULL", value);
}

std::string MakeU64ArrayLiteral(const Shape& values) {
    std::vector<std::string> formattedValues;
    formattedValues.reserve(values.size());

    for (const auto& value : values) {
        formattedValues.push_back(MakeU64Literal(value));
    }

    return jst::fmt::format("{{{}}}", jst::fmt::join(formattedValues, ", "));
}

std::string BuildKernelConstants(const Tensor& input) {
    auto constants = jst::fmt::format(
        "static constexpr unsigned long long kElementCount = {};\n",
        MakeU64Literal(input.size())
    );

    if (!input.contiguous()) {
        constants += jst::fmt::format(
            "static constexpr int kRank = {};\n"
            "static constexpr unsigned long long kShape[] = {};\n"
            "static constexpr unsigned long long kStride[] = {};\n",
            input.rank(),
            MakeU64ArrayLiteral(input.shape()),
            MakeU64ArrayLiteral(input.stride())
        );
    }

    return constants;
}

const char* ScalarTypeName(const DataType& dtype) {
    switch (dtype) {
        case DataType::I8:
            return "signed char";
        case DataType::U8:
            return "unsigned char";
        case DataType::I16:
            return "short";
        case DataType::U16:
            return "unsigned short";
        case DataType::I32:
            return "int";
        case DataType::U32:
            return "unsigned int";
        default:
            return nullptr;
    }
}

const char* ComplexTypeName(const DataType& dtype) {
    switch (dtype) {
        case DataType::CI8:
            return "KernelComplex<signed char>";
        case DataType::CU8:
            return "KernelComplex<unsigned char>";
        case DataType::CI16:
            return "KernelComplex<short>";
        case DataType::CU16:
            return "KernelComplex<unsigned short>";
        case DataType::CI32:
            return "KernelComplex<int>";
        case DataType::CU32:
            return "KernelComplex<unsigned int>";
        default:
            return nullptr;
    }
}

const char* ReciprocalExpression(const DataType& dtype) {
    switch (dtype) {
        case DataType::I8:
        case DataType::U8:
        case DataType::CI8:
        case DataType::CU8:
            return "(1.0f / 128.0f)";
        case DataType::I16:
        case DataType::U16:
        case DataType::CI16:
        case DataType::CU16:
            return "(1.0f / 32768.0f)";
        case DataType::I32:
        case DataType::U32:
        case DataType::CI32:
        case DataType::CU32:
            return "(1.0f / 2147483648.0f)";
        default:
            return nullptr;
    }
}

Result BuildConversionDecls(const DataType& inputDtype,
                            const DataType& outputDtype,
                            std::string& conversionDecls) {
    if (outputDtype == DataType::F32) {
        const char* inputType = ScalarTypeName(inputDtype);
        const char* reciprocal = ReciprocalExpression(inputDtype);
        if (!inputType || !reciprocal) {
            JST_ERROR("[MODULE_CAST_NATIVE_CUDA] Unsupported conversion '{}' -> '{}'.",
                      inputDtype, outputDtype);
            return Result::ERROR;
        }

        conversionDecls = jst::fmt::format(
            "using InputValue = {};\n"
            "using OutputValue = float;\n\n"
            "__device__ __forceinline__ OutputValue ConvertValue(const InputValue in) {{\n"
            "    return static_cast<float>(in) * {};\n"
            "}}\n",
            inputType,
            reciprocal
        );

        return Result::SUCCESS;
    }

    if (outputDtype == DataType::CF32) {
        const char* inputType = ComplexTypeName(inputDtype);
        if (!inputType) {
            JST_ERROR("[MODULE_CAST_NATIVE_CUDA] Unsupported conversion '{}' -> '{}'.",
                      inputDtype, outputDtype);
            return Result::ERROR;
        }

        const char* reciprocal = ReciprocalExpression(inputDtype);

        conversionDecls = jst::fmt::format(
            "using InputValue = {};\n"
            "using OutputValue = KernelComplex<float>;\n\n"
            "__device__ __forceinline__ OutputValue ConvertValue(const InputValue in) {{\n"
            "    OutputValue out;\n"
            "    out.real = static_cast<float>(in.real) * {};\n"
            "    out.imag = static_cast<float>(in.imag) * {};\n"
            "    return out;\n"
            "}}\n",
            inputType,
            reciprocal,
            reciprocal
        );
        return Result::SUCCESS;
    }

    JST_ERROR("[MODULE_CAST_NATIVE_CUDA] Unsupported conversion '{}' -> '{}'.",
              inputDtype, outputDtype);
    return Result::ERROR;
}

}  // namespace

struct CastImplNativeCuda : public CastImpl,
                            public NativeCudaRuntimeContext,
                            public Scheduler::Context {
 public:
    Result create() final;

    Result computeInitialize() override;
    Result computeSubmit(const cudaStream_t& stream) override;
    Result computeDeinitialize() override;

 private:
    bool kernelCreated = false;
    std::string kernelSource;
    std::unordered_map<std::string, std::string> kernelPieces;
};

Result CastImplNativeCuda::create() {
    JST_CHECK(CastImpl::create());

    if (bypass) {
        return Result::SUCCESS;
    }

    kernelSource = input.contiguous() ? kCastContiguousKernelSource : kCastStridedKernelSource;

    kernelPieces.clear();
    kernelPieces["KERNEL_CONSTANTS"] = BuildKernelConstants(input);
    JST_CHECK(BuildConversionDecls(input.dtype(), outputDtype, kernelPieces["CONVERSION_DECLS"]));

    return Result::SUCCESS;
}

Result CastImplNativeCuda::computeInitialize() {
    if (bypass) {
        return Result::SUCCESS;
    }

    JST_CHECK(createKernel(kCastKernelName, kernelSource, kernelPieces));
    kernelCreated = true;

    return Result::SUCCESS;
}

Result CastImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    if (bypass) {
        return Result::SUCCESS;
    }

    if (output.size() == 0) {
        return Result::SUCCESS;
    }

    const auto* inputBase = static_cast<const std::uint8_t*>(input.buffer().data());
    void* outputData = output.buffer().data();

    if (!inputBase || !outputData) {
        JST_ERROR("[MODULE_CAST_NATIVE_CUDA] Missing input or output device buffer.");
        return Result::ERROR;
    }

    const void* inputData = inputBase + input.offsetBytes();
    void* inputArgument = const_cast<void*>(inputData);
    void* outputArgument = outputData;

    void* arguments[] = {
        &inputArgument,
        &outputArgument,
    };

    const Extent3D<U64> block = {kThreadsPerBlock, 1, 1};
    const Extent3D<U64> grid = {(output.size() + kThreadsPerBlock - 1) / kThreadsPerBlock, 1, 1};

    return scheduleKernel(kCastKernelName, stream, grid, block, arguments);
}

Result CastImplNativeCuda::computeDeinitialize() {
    if (kernelCreated) {
        JST_CHECK(destroyKernel(kCastKernelName));
    }

    kernelCreated = false;

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(CastImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
