#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/backend/base.hh>
#include <jetstream/backend/devices/cuda/helpers.hh>

#include <unordered_set>

namespace Jetstream {

namespace {

std::string IndentKernelPiece(const std::string& piece, const std::string& indentation) {
    std::string indentedPiece;

    size_t lineStart = 0;
    while (lineStart < piece.size()) {
        const size_t lineEnd = piece.find('\n', lineStart);
        const bool hasNewline = lineEnd != std::string::npos;

        std::string line = hasNewline ? piece.substr(lineStart, lineEnd - lineStart)
                                      : piece.substr(lineStart);

        const bool hasCarriageReturn = !line.empty() && line.back() == '\r';
        if (hasCarriageReturn) {
            line.pop_back();
        }

        if (!line.empty()) {
            indentedPiece += indentation;
        }

        indentedPiece += line;

        if (!hasNewline) {
            break;
        }

        indentedPiece += hasCarriageReturn ? "\r\n" : "\n";
        lineStart = lineEnd + 1;
    }

    return indentedPiece;
}

Result ExpandKernelPieces(const std::string& sourceTemplate,
                          const std::unordered_map<std::string, std::string>& pieces,
                          std::string& expandedSource) {
    expandedSource.clear();

    std::unordered_set<std::string> usedPieces;

    size_t lineStart = 0;
    while (lineStart < sourceTemplate.size()) {
        const size_t lineEnd = sourceTemplate.find('\n', lineStart);
        const bool hasNewline = lineEnd != std::string::npos;

        const std::string rawLine = hasNewline ? sourceTemplate.substr(lineStart, lineEnd - lineStart)
                                               : sourceTemplate.substr(lineStart);

        std::string line = rawLine;
        const bool hasCarriageReturn = !line.empty() && line.back() == '\r';
        if (hasCarriageReturn) {
            line.pop_back();
        }

        size_t contentStart = 0;
        while (contentStart < line.size() && (line[contentStart] == ' ' || line[contentStart] == '\t')) {
            contentStart++;
        }

        const std::string indentation = line.substr(0, contentStart);

        static constexpr const char* markerPrefix = "<<<";
        static constexpr const char* markerSuffix = ">>>";

        if (line.compare(contentStart, std::char_traits<char>::length(markerPrefix), markerPrefix) == 0) {
            const size_t pieceStart = contentStart + std::char_traits<char>::length(markerPrefix);
            const size_t pieceEnd = line.find(markerSuffix, pieceStart);
            const bool isPlaceholder = pieceEnd != std::string::npos &&
                                       line.find_first_not_of(" \t",
                                                              pieceEnd + std::char_traits<char>::length(markerSuffix)) == std::string::npos;

            if (isPlaceholder) {
                const std::string pieceId = line.substr(pieceStart, pieceEnd - pieceStart);
                const auto piece = pieces.find(pieceId);

                if (piece == pieces.end()) {
                    JST_ERROR("[RUNTIME_CONTEXT_NATIVE_CUDA] Kernel source references missing piece '{}'.", pieceId);
                    return Result::ERROR;
                }

                usedPieces.insert(pieceId);

                const std::string indentedPiece = IndentKernelPiece(piece->second, indentation);
                expandedSource += indentedPiece;

                if (hasNewline && (indentedPiece.empty() || indentedPiece.back() != '\n')) {
                    expandedSource += hasCarriageReturn ? "\r\n" : "\n";
                }

                lineStart = hasNewline ? lineEnd + 1 : sourceTemplate.size();
                continue;
            }
        }

        expandedSource += rawLine;

        if (hasNewline) {
            expandedSource += '\n';
        }

        lineStart = hasNewline ? lineEnd + 1 : sourceTemplate.size();
    }

    for (const auto& piece : pieces) {
        if (!usedPieces.contains(piece.first)) {
            JST_ERROR("[RUNTIME_CONTEXT_NATIVE_CUDA] Kernel piece '{}' was provided but not used in the source template.", piece.first);
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

}  // namespace

struct NativeCudaRuntimeContext::Impl {
    struct KernelState {
        CUfunction function;
        CUmodule module;
    };

    std::unordered_map<std::string, KernelState> kernels;
};

NativeCudaRuntimeContext::NativeCudaRuntimeContext() {
    pimpl = std::make_unique<Impl>();
}

NativeCudaRuntimeContext::~NativeCudaRuntimeContext() {
    pimpl.reset();
}

Result NativeCudaRuntimeContext::createKernel(const std::string& name,
                                              const std::string& source,
                                              const std::unordered_map<std::string, std::string>& pieces,
                                              const std::vector<std::string>& headers) {
    if (pimpl->kernels.contains(name)) {
        JST_ERROR("[RUNTIME_CONTEXT_NATIVE_CUDA] Kernel name '{}' already exists.", name);
        return Result::ERROR;
    }

    Impl::KernelState kernel;

    std::string expandedSource = source;
    if (!pieces.empty()) {
        JST_CHECK(ExpandKernelPieces(source, pieces, expandedSource));
        JST_TRACE("[RUNTIME_CONTEXT_NATIVE_CUDA] Expanded kernel source ({}):\n{}", name, expandedSource);
    }

    // Create program.

    nvrtcProgram program;
    JST_NVRTC_CHECK(nvrtcCreateProgram(&program, expandedSource.c_str(), nullptr, 0, nullptr, nullptr), [&]{
        JST_ERROR("[RUNTIME_CONTEXT_NATIVE_CUDA] Can't create program: {}", err);
    });

    // Load headers into program.

    (void)headers;  // TODO: Implement header loading.

    // Add name expression.

    JST_NVRTC_CHECK(nvrtcAddNameExpression(program, name.c_str()), [&]{
        JST_ERROR("[RUNTIME_CONTEXT_NATIVE_CUDA] Can't add name expression: {}", err);
    });

    // Compile program.

    std::string cc = Backend::State<DeviceType::CUDA>()->getComputeCapability();
    std::string arch = jst::fmt::format("--gpu-architecture=compute_{}", cc);

    const std::vector<const char*> options = {
        arch.c_str(),
        "--std=c++20"
    };

    JST_TRACE("[RUNTIME_CONTEXT_NATIVE_CUDA] Compiling kernel with options: {}", options);

    JST_NVRTC_CHECK(nvrtcCompileProgram(program, options.size(), options.data()), [&]{
        size_t logSize;
        nvrtcGetProgramLogSize(program, &logSize);
        std::vector<char> log(logSize);
        nvrtcGetProgramLog(program, log.data());
        JST_ERROR("[RUNTIME_CONTEXT_NATIVE_CUDA] Can't compile program:\n{}", log.data());
    });

    // Get PTX.

    size_t ptxSize;
    JST_NVRTC_CHECK(nvrtcGetPTXSize(program, &ptxSize), [&]{
        JST_ERROR("[RUNTIME_CONTEXT_NATIVE_CUDA] Can't get PTX size: {}", err);
    });

    std::vector<char> ptx(ptxSize);
    JST_NVRTC_CHECK(nvrtcGetPTX(program, ptx.data()), [&]{
        JST_ERROR("[RUNTIME_CONTEXT_NATIVE_CUDA] Can't get PTX: {}", err);
    });

    // Print PTX.

    JST_TRACE("[RUNTIME_CONTEXT_NATIVE_CUDA] Generated kernel PTX ({}):\n{}", name, ptx.data());

    // Get lowered name.

    const char* loweredName;
    JST_NVRTC_CHECK(nvrtcGetLoweredName(program, name.c_str(), &loweredName), [&]{
        JST_ERROR("[RUNTIME_CONTEXT_NATIVE_CUDA] Can't get lowered name: {}", err);
    });

    // Create module.

    JST_CUDA_CHECK(cuModuleLoadData(&kernel.module, ptx.data()), [&]{
        JST_ERROR("[RUNTIME_CONTEXT_NATIVE_CUDA] Can't load module: {}", err);
    });

    // Get function.

    JST_CUDA_CHECK(cuModuleGetFunction(&kernel.function, kernel.module, loweredName), [&]{
        JST_ERROR("[RUNTIME_CONTEXT_NATIVE_CUDA] Can't get function: {}", err);
    });

    // Destroy program.

    JST_NVRTC_CHECK(nvrtcDestroyProgram(&program), [&]{
        JST_ERROR("[RUNTIME_CONTEXT_NATIVE_CUDA] Can't destroy program: {}", err);
    });

    // Store kernel.

    pimpl->kernels[name] = kernel;

    return Result::SUCCESS;
}

Result NativeCudaRuntimeContext::createKernelFromPtx(const std::string& name,
                                                     const std::string& ptx,
                                                     const std::string& kernelName) {
    if (pimpl->kernels.contains(name)) {
        JST_ERROR("[RUNTIME_CONTEXT_NATIVE_CUDA] Kernel name '{}' already exists.", name);
        return Result::ERROR;
    }

    Impl::KernelState kernel;

    // Create module.

    JST_CUDA_CHECK(cuModuleLoadData(&kernel.module, ptx.data()), [&]{
        JST_ERROR("[RUNTIME_CONTEXT_NATIVE_CUDA] Can't load module: {}", err);
    });

    // Get function.

    JST_CUDA_CHECK(cuModuleGetFunction(&kernel.function, kernel.module, kernelName.c_str()), [&]{
        JST_ERROR("[RUNTIME_CONTEXT_NATIVE_CUDA] Can't get function: {}", err);
    });

    // Store kernel.

    pimpl->kernels[name] = kernel;

    return Result::SUCCESS;
}

Result NativeCudaRuntimeContext::scheduleKernel(const std::string& name,
                                                const cudaStream_t& stream,
                                                const Extent3D<U64>& grid,
                                                const Extent3D<U64>& block,
                                                void** arguments) {
    if (!pimpl->kernels.contains(name)) {
        JST_ERROR("[RUNTIME_CONTEXT_NATIVE_CUDA] Kernel name '{}' doesn't exist.", name);
        return Result::ERROR;
    }

    auto& kernel = pimpl->kernels[name];

    JST_CUDA_CHECK(cuLaunchKernel(kernel.function,
                                  grid.x, grid.y, grid.z,
                                  block.x, block.y, block.z,
                                  0, stream, arguments, 0), [&]{
        JST_ERROR("[CUDA] Can't launch kernel: {}", err);
    });

    return Result::SUCCESS;
}

Result NativeCudaRuntimeContext::destroyKernel(const std::string& name) {
    if (!pimpl->kernels.contains(name)) {
        JST_ERROR("[RUNTIME_CONTEXT_NATIVE_CUDA] Kernel name '{}' doesn't exist.", name);
        return Result::ERROR;
    }

    auto& kernel = pimpl->kernels[name];

    JST_CUDA_CHECK(cuModuleUnload(kernel.module), [&]{
        JST_ERROR("[RUNTIME_CONTEXT_NATIVE_CUDA] Can't unload module: {}", err);
    });

    pimpl->kernels.erase(name);

    return Result::SUCCESS;
}

Result NativeCudaRuntimeContext::computeInitialize() {
    return Result::SUCCESS;
}

Result NativeCudaRuntimeContext::computeSubmit(const cudaStream_t&) {
    return Result::SUCCESS;
}

Result NativeCudaRuntimeContext::computeDeinitialize() {
    return Result::SUCCESS;
}

}  // namespace Jetstream
