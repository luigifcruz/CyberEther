#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/backend/base.hh>
#include <jetstream/backend/devices/cuda/helpers.hh>

namespace Jetstream {

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
                                              const std::vector<std::string>& headers) {
    if (pimpl->kernels.contains(name)) {
        JST_ERROR("[RUNTIME_CONTEXT_NATIVE_CUDA] Kernel name '{}' already exists.", name);
        return Result::ERROR;
    }

    Impl::KernelState kernel;

    // Create program.

    nvrtcProgram program;
    JST_NVRTC_CHECK(nvrtcCreateProgram(&program, source.c_str(), nullptr, 0, nullptr, nullptr), [&]{
        JST_ERROR("[RUNTIME_CONTEXT_NATIVE_CUDA] Can't create program: {}", err);
    });

    // Load headers into program.

    // TODO: Implement header loading.

    // Add name expression.

    JST_NVRTC_CHECK(nvrtcAddNameExpression(program, name.c_str()), [&]{
        JST_ERROR("[RUNTIME_CONTEXT_NATIVE_CUDA] Can't add name expression: {}", err);
    });

    // Compile program.

    std::string cc = Backend::State<DeviceType::CUDA>()->getComputeCapability();
    std::string arch = jst::fmt::format("--gpu-architecture=compute_{}", cc);

    const std::vector<const char*> options = {
        arch.c_str(),
        "--std=c++14"
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
