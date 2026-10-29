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
constexpr const char* kCastKernelName = "cast_kernel";

constexpr const char* kCastContiguousKernelSource = R"(
template<typename T>
struct alignas(sizeof(T) * 2) KernelComplex {
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
struct alignas(sizeof(T) * 2) KernelComplex {
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
        case DataType::F32:
        case DataType::CF32:
            return "float";
        case DataType::F64:
        case DataType::CF64:
            return "double";
        case DataType::I8:
        case DataType::CI8:
            return "signed char";
        case DataType::U8:
        case DataType::CU8:
            return "unsigned char";
        case DataType::I16:
        case DataType::CI16:
            return "short";
        case DataType::U16:
        case DataType::CU16:
            return "unsigned short";
        case DataType::I32:
        case DataType::CI32:
            return "int";
        case DataType::U32:
        case DataType::CU32:
            return "unsigned int";
        case DataType::I64:
        case DataType::CI64:
            return "long long";
        case DataType::U64:
        case DataType::CU64:
            return "unsigned long long";
        default:
            return nullptr;
    }
}

bool NeedsDoubleWork(const DataType& dtype) {
    switch (dtype) {
        case DataType::F64:
        case DataType::CF64:
        case DataType::I32:
        case DataType::CI32:
        case DataType::U32:
        case DataType::CU32:
        case DataType::I64:
        case DataType::CI64:
        case DataType::U64:
        case DataType::CU64:
            return true;
        default:
            return false;
    }
}

std::string MakeFloatLiteral(const F64 value, const bool doublePrecision) {
    std::string literal = jst::fmt::format("{:.17g}", value);
    if (literal.find_first_of(".e") == std::string::npos) {
        literal += ".0";
    }
    return doublePrecision ? literal : literal + "f";
}

std::string BuildConversionDecls(const DataType& inputDtype,
                                 const DataType& outputDtype,
                                 const CastImpl::Scale& inputScale,
                                 const CastImpl::Scale& outputScale) {
    const bool doubleWork = NeedsDoubleWork(inputDtype) || NeedsDoubleWork(outputDtype);
    const bool doubleOutput = outputDtype == DataType::F64 || outputDtype == DataType::CF64;
    const bool inputComplex = IsDataTypeComplex(inputDtype);
    const bool outputComplex = IsDataTypeComplex(outputDtype);

    std::string decls = jst::fmt::format(
        "using Work = {};\n"
        "using InputScalar = {};\n"
        "using OutputScalar = {};\n"
        "using InputValue = {};\n"
        "using OutputValue = {};\n\n",
        doubleWork ? "double" : "float",
        ScalarTypeName(inputDtype),
        ScalarTypeName(outputDtype),
        inputComplex ? "KernelComplex<InputScalar>" : "InputScalar",
        outputComplex ? "KernelComplex<OutputScalar>" : "OutputScalar"
    );

    if (inputScale.integer) {
        decls += jst::fmt::format(
            "__device__ __forceinline__ long long Decode(const InputScalar value) {{\n"
            "    return {};\n"
            "}}\n\n",
            inputScale.isUnsigned ?
                jst::fmt::format("static_cast<long long>(static_cast<unsigned long long>(value) - {}ULL)",
                                 U64{1} << (inputScale.bits - 1)) :
                std::string("static_cast<long long>(value)")
        );
    }

    if (outputScale.integer) {
        decls += jst::fmt::format(
            "__device__ __forceinline__ OutputScalar Encode(const long long value) {{\n"
            "    return {};\n"
            "}}\n\n",
            outputScale.isUnsigned ?
                jst::fmt::format("static_cast<OutputScalar>(static_cast<unsigned long long>(value) + {}ULL)",
                                 U64{1} << (outputScale.bits - 1)) :
                std::string("static_cast<OutputScalar>(value)")
        );
    }

    decls += "__device__ __forceinline__ OutputScalar ConvertScalar(const InputScalar value) {\n";

    if (!inputScale.integer && !outputScale.integer) {
        decls += "    return static_cast<OutputScalar>(value);\n";
    } else if (!outputScale.integer) {
        decls += jst::fmt::format(
            "    return static_cast<OutputScalar>(Decode(value)) * {};\n",
            MakeFloatLiteral(1.0 / inputScale.fullScale, doubleOutput)
        );
    } else if (!inputScale.integer) {
        const I64 outputMax = std::numeric_limits<I64>::max() >> (64 - outputScale.bits);
        const std::string gain = MakeFloatLiteral(outputScale.fullScale, doubleWork);
        decls += jst::fmt::format(
            "    Work level = static_cast<Work>(value);\n"
            "    if (level != level) {{\n"
            "        level = 0;\n"
            "    }}\n"
            "    const Work scaled = {}(level * {});\n"
            "    if (scaled >= {}) {{\n"
            "        return Encode({}LL);\n"
            "    }}\n"
            "    if (scaled < -{}) {{\n"
            "        return Encode(-{}LL - 1LL);\n"
            "    }}\n"
            "    return Encode(static_cast<long long>(scaled));\n",
            doubleWork ? "rint" : "rintf",
            gain, gain, outputMax, gain, outputMax
        );
    } else {
        const I64 shift = static_cast<I64>(outputScale.bits) - static_cast<I64>(inputScale.bits);
        if (shift >= 0) {
            decls += jst::fmt::format(
                "    return Encode(Decode(value) * {}LL);\n",
                I64{1} << shift
            );
        } else {
            const I64 outputMax = std::numeric_limits<I64>::max() >> (64 - outputScale.bits);
            decls += jst::fmt::format(
                "    const long long level = Decode(value);\n"
                "    long long quotient = level >> {};\n"
                "    const long long remainder = level - quotient * {}LL;\n"
                "    if (remainder > {}LL || (remainder == {}LL && (quotient & 1LL))) {{\n"
                "        quotient += 1LL;\n"
                "    }}\n"
                "    return Encode(quotient < {}LL ? quotient : {}LL);\n",
                -shift, I64{1} << -shift, I64{1} << (-shift - 1), I64{1} << (-shift - 1),
                outputMax, outputMax
            );
        }
    }

    decls += "}\n\n";

    if (inputComplex) {
        decls +=
            "__device__ __forceinline__ OutputValue ConvertValue(const InputValue in) {\n"
            "    OutputValue out;\n"
            "    out.real = ConvertScalar(in.real);\n"
            "    out.imag = ConvertScalar(in.imag);\n"
            "    return out;\n"
            "}\n";
    } else if (outputComplex) {
        decls += jst::fmt::format(
            "__device__ __forceinline__ OutputValue ConvertValue(const InputValue in) {{\n"
            "    OutputValue out;\n"
            "    out.real = ConvertScalar(in);\n"
            "    out.imag = {};\n"
            "    return out;\n"
            "}}\n",
            outputScale.integer ? "Encode(0LL)" : "static_cast<OutputScalar>(0)"
        );
    } else {
        decls +=
            "__device__ __forceinline__ OutputValue ConvertValue(const InputValue in) {\n"
            "    return ConvertScalar(in);\n"
            "}\n";
    }

    return decls;
}

}  // namespace

struct CastImplNativeCuda : public CastImpl,
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
    std::string kernelSource;
    std::unordered_map<std::string, std::string> kernelPieces;
};

Result CastImplNativeCuda::validate() {
    JST_CHECK(CastImpl::validate());

    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& candidateInput = inputs().at("buffer").tensor;
    if (!candidateInput.validShape() || candidateInput.size() == 0) {
        return Result::SUCCESS;
    }

    if (validatedBypass) {
        return Result::SUCCESS;
    }

    const U64 blockCount = validatedOutputElementCount / kThreadsPerBlock +
                           (validatedOutputElementCount % kThreadsPerBlock != 0);
    if (blockCount > kMaxGridSizeX) {
        JST_ERROR("[MODULE_CAST_NATIVE_CUDA] Output size exceeds the CUDA grid limit.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result CastImplNativeCuda::create() {
    JST_CHECK(CastImpl::create());

    if (bypass) {
        return Result::SUCCESS;
    }

    kernelSource = input.contiguous() ? kCastContiguousKernelSource : kCastStridedKernelSource;

    kernelPieces.clear();
    kernelPieces["KERNEL_CONSTANTS"] = BuildKernelConstants(input);
    kernelPieces["CONVERSION_DECLS"] = BuildConversionDecls(input.dtype(), outputDtype,
                                                         inputScale, outputScale);

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
