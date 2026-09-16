#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>

#include <jetstream/memory/macros.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

constexpr U64 kThreadsPerBlock = 256;
constexpr U64 kMaxGridSizeX = std::numeric_limits<I32>::max();
constexpr const char* kRemoveIndicesKernelName = "remove_indices_kernel";
constexpr const char* kRemoveIndicesKernelSource = R"(
<<<KERNEL_CONSTANTS>>>
extern "C" __global__ void remove_indices_kernel(
    const CopyWord* input,
    CopyWord* output,
    const unsigned long long* keptIndices) {
    const unsigned long long index =
        (static_cast<unsigned long long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (index >= kWordCount) {
        return;
    }

    unsigned long long sourceIndex = 0;
    if constexpr (kContiguous) {
        const unsigned long long inner = index % kInnerWords;
        const unsigned long long axisIndex = (index / kInnerWords) % kOutputAxisSize;
        const unsigned long long outer = index / (kInnerWords * kOutputAxisSize);
        sourceIndex = (outer * kInputAxisSize + keptIndices[axisIndex]) * kInnerWords + inner;
    } else {
        unsigned long long remaining = index / kWordsPerElement;
        unsigned long long sourceElement = 0;
        for (int axis = kRank - 1; axis >= 0; --axis) {
            unsigned long long coordinate = remaining % kShape[axis];
            remaining /= kShape[axis];
            if (axis == kAxis) {
                coordinate = keptIndices[coordinate];
            }
            sourceElement += coordinate * kInputStride[axis];
        }
        sourceIndex = sourceElement * kWordsPerElement + index % kWordsPerElement;
    }

    output[index] = input[sourceIndex];
}
)";

U64 CopyWordSize(const Tensor& tensor) {
    return std::min(tensor.elementSize(), static_cast<U64>(sizeof(U64)));
}

std::string MakeU64ArrayLiteral(const Shape& values) {
    std::vector<std::string> formattedValues;
    formattedValues.reserve(values.size());
    for (const auto value : values) {
        formattedValues.push_back(jst::fmt::format("{}ULL", value));
    }
    return jst::fmt::format("{{{}}}", jst::fmt::join(formattedValues, ", "));
}

std::string BuildKernelConstants(const Tensor& input,
                                 const Tensor& output,
                                 const Index axis) {
    const U64 wordSize = CopyWordSize(input);
    const char* wordType = "unsigned long long";
    switch (wordSize) {
        case 1: wordType = "unsigned char"; break;
        case 2: wordType = "unsigned short"; break;
        case 4: wordType = "unsigned int"; break;
        default: break;
    }

    // Copy integer words to preserve every dtype bit-for-bit, with adjacent
    // threads accessing adjacent words even for 16-byte complex elements.
    const U64 wordsPerElement = input.elementSize() / wordSize;
    return jst::fmt::format(
        "using CopyWord = {};\n"
        "static constexpr unsigned long long kWordCount = {}ULL;\n"
        "static constexpr unsigned long long kWordsPerElement = {}ULL;\n"
        "static constexpr bool kContiguous = {};\n"
        "static constexpr unsigned long long kInnerWords = {}ULL;\n"
        "static constexpr unsigned long long kInputAxisSize = {}ULL;\n"
        "static constexpr unsigned long long kOutputAxisSize = {}ULL;\n"
        "static constexpr int kRank = {};\n"
        "static constexpr int kAxis = {};\n"
        "static constexpr unsigned long long kShape[] = {};\n"
        "static constexpr unsigned long long kInputStride[] = {};\n",
        wordType,
        output.sizeBytes() / wordSize,
        wordsPerElement,
        input.contiguous(),
        output.stride(axis) * wordsPerElement,
        input.shape(axis),
        output.shape(axis),
        input.rank(),
        axis,
        MakeU64ArrayLiteral(output.shape()),
        MakeU64ArrayLiteral(input.stride())
    );
}

}  // namespace

struct RemoveIndicesImplNativeCuda : public RemoveIndicesImpl,
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
    Tensor indicesTensor;
    U64 wordCount = 0;
    std::unordered_map<std::string, std::string> kernelPieces;
};

Result RemoveIndicesImplNativeCuda::validate() {
    JST_CHECK(RemoveIndicesImpl::validate());

    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }
    const Tensor& inputTensor = inputs().at("buffer").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.rank() > static_cast<Index>(std::numeric_limits<I32>::max())) {
        JST_ERROR("[MODULE_REMOVE_INDICES_NATIVE_CUDA] Tensor rank exceeds the CUDA kernel limit.");
        return Result::ERROR;
    }

    const U64 axisSize = inputTensor.shape(validatedResolvedAxis);
    const U64 keptCount = axisSize - validatedRemovedIndices.size();
    const U64 outputElements = (inputTensor.size() / axisSize) * keptCount;
    const U64 candidateWordCount = outputElements *
                                    (inputTensor.elementSize() / CopyWordSize(inputTensor));
    const U64 blockCount = candidateWordCount / kThreadsPerBlock +
                           (candidateWordCount % kThreadsPerBlock != 0);
    if (blockCount > kMaxGridSizeX) {
        JST_ERROR("[MODULE_REMOVE_INDICES_NATIVE_CUDA] Output size exceeds the CUDA grid limit.");
        return Result::ERROR;
    }

    U64 metadataBytes = 0;
    U64 alignedBytes = 0;
    if (!detail::CheckedMultiply(keptCount, static_cast<U64>(sizeof(U64)), metadataBytes) ||
        !detail::CheckedPageAlignedSize(metadataBytes, alignedBytes) ||
        !detail::CheckedPageAlignedSize(outputElements * inputTensor.elementSize(), alignedBytes)) {
        JST_ERROR("[MODULE_REMOVE_INDICES_NATIVE_CUDA] Allocation size is too large.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result RemoveIndicesImplNativeCuda::create() {
    JST_CHECK(RemoveIndicesImpl::create());
    wordCount = output.sizeBytes() / CopyWordSize(input);
    kernelPieces["KERNEL_CONSTANTS"] = BuildKernelConstants(input, output, resolvedAxis);
    return Result::SUCCESS;
}

Result RemoveIndicesImplNativeCuda::computeInitialize() {
    Buffer::Config metadataConfig{};
    metadataConfig.hostAccessible = true;
    JST_CHECK(indicesTensor.create(device(), DataType::U64,
                                    {static_cast<U64>(keptIndices.size())}, metadataConfig));
    std::copy(keptIndices.begin(), keptIndices.end(), indicesTensor.data<U64>());

    JST_CHECK(createKernel(kRemoveIndicesKernelName,
                           kRemoveIndicesKernelSource, kernelPieces));
    kernelCreated = true;
    return Result::SUCCESS;
}

Result RemoveIndicesImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    const auto* inputBase = static_cast<const U8*>(input.buffer().data());
    auto* outputBase = static_cast<U8*>(output.buffer().data());
    void* indicesData = indicesTensor.data();
    if (!inputBase || !outputBase || !indicesData) {
        JST_ERROR("[MODULE_REMOVE_INDICES_NATIVE_CUDA] Missing device buffer.");
        return Result::ERROR;
    }

    // CUDA tensor data pointers refer to the backing buffer; apply view offsets
    // explicitly so sliced tensors read from the correct starting element.
    const void* inputData = inputBase + input.offsetBytes();
    void* inputArgument = const_cast<void*>(inputData);
    void* outputData = outputBase + output.offsetBytes();
    void* arguments[] = {&inputArgument, &outputData, &indicesData};

    const Extent3D<U64> block = {kThreadsPerBlock, 1, 1};
    const Extent3D<U64> grid = {
        wordCount / kThreadsPerBlock + (wordCount % kThreadsPerBlock != 0),
        1,
        1,
    };
    return scheduleKernel(kRemoveIndicesKernelName, stream, grid, block, arguments);
}

Result RemoveIndicesImplNativeCuda::computeDeinitialize() {
    if (kernelCreated) {
        JST_CHECK(destroyKernel(kRemoveIndicesKernelName));
    }
    kernelCreated = false;
    indicesTensor = {};
    return Result::SUCCESS;
}

JST_REGISTER_MODULE(RemoveIndicesImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
