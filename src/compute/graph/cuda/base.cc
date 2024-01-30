#include "jetstream/compute/graph/cuda.hh"
#include "jetstream/backend/devices/cuda/helpers.hh"

#include <nvrtc.h>

namespace Jetstream {

struct CUDA::Impl {
    struct Kernel {
        CUfunction function;
        CUmodule module;
    };

    std::unordered_map<std::string, Kernel> kernels;
};

CUDA::CUDA() {
    context = std::make_shared<Compute::Context>();
    context->cuda = this;

    pimpl = std::make_unique<Impl>();
}

CUDA::~CUDA() {
    context.reset();
    pimpl.reset();
}

Result CUDA::create() {
    JST_DEBUG("Creating new CUDA compute graph.");

    // Create CUDA stream.

    JST_CUDA_CHECK(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking), [&]{
        JST_ERROR("[CUDA] Can't create stream: {}", err);
    });

    // Create blocks.

    for (const auto& block : blocks) {
        JST_CHECK(block->createCompute(*context));
    }

    return Result::SUCCESS;
}

Result CUDA::computeReady() {
    for (const auto& block : blocks) {
        JST_CHECK(block->computeReady());
    }
    return Result::SUCCESS;
}

Result CUDA::compute() {
    // Execute blocks.

    for (const auto& block : blocks) {
        JST_CHECK(block->compute(*context));

        // Check for CUDA errors.

        JST_CUDA_CHECK(cudaGetLastError(), [&]{
            JST_ERROR("[CUDA] Module kernel execution failed: {}", err);
        });
    }

    // Wait for all blocks to finish.

    JST_CUDA_CHECK(cudaStreamSynchronize(_stream), [&]{
        JST_ERROR("[CUDA] Can't synchronize stream: {}", err);
    });

    return Result::SUCCESS;
}

Result CUDA::destroy() {
    // Destroy blocks.

    for (const auto& block : blocks) {
        JST_CHECK(block->destroyCompute(*context));
    }
    blocks.clear();

    // Destroy kernels.

    std::vector<std::string> kernel_names;
    for (const auto& [name, _] : pimpl->kernels) {
        kernel_names.push_back(name);
    }
    for (const auto& name : kernel_names) {
        JST_CHECK(destroyKernel(name));
    }

    // Destroy CUDA stream.

    JST_CUDA_CHECK(cudaStreamDestroy(_stream), [&]{
        JST_ERROR("[CUDA] Can't destroy stream: {}", err);
    });

    return Result::SUCCESS;
}

Result CUDA::createKernel(const std::string& name, 
                          const std::string& source,
                          const std::vector<KernelHeader>& headers) {
    if (pimpl->kernels.contains(name)) {
        JST_ERROR("[CUDA] Kernel with name '{}' already exists.", name);
    }
    auto& kernel = pimpl->kernels[name];

    // Create program.

    nvrtcProgram program;
    JST_NVRTC_CHECK(nvrtcCreateProgram(&program, source.c_str(), nullptr, 0, nullptr, nullptr), [&]{
        JST_ERROR("[CUDA] Can't create program: {}", err);
    });

    // Add name expression.

    JST_NVRTC_CHECK(nvrtcAddNameExpression(program, name.c_str()), [&]{
        JST_ERROR("[CUDA] Can't add name expression: {}", err);
    });

    // Compile program.

    const std::vector<const char*> options = {
        "--gpu-architecture=compute_86",
        "--std=c++14"
    };

    JST_NVRTC_CHECK(nvrtcCompileProgram(program, options.size(), options.data()), [&]{
        size_t logSize;
        nvrtcGetProgramLogSize(program, &logSize);
        std::vector<char> log(logSize);
        nvrtcGetProgramLog(program, log.data());
        JST_ERROR("[CUDA] Can't compile program:\n{}", log.data());
    });

    // Get PTX.

    size_t ptxSize;
    JST_NVRTC_CHECK(nvrtcGetPTXSize(program, &ptxSize), [&]{
        JST_ERROR("[CUDA] Can't get PTX size: {}", err);
    });

    std::vector<char> ptx(ptxSize);
    JST_NVRTC_CHECK(nvrtcGetPTX(program, ptx.data()), [&]{
        JST_ERROR("[CUDA] Can't get PTX: {}", err);
    });

    // Print PTX.

    JST_TRACE("[CUDA] Generated kernel PTX ({}):\n{}", name, ptx.data());

    // Get lowered name.

    const char* loweredName;
    JST_NVRTC_CHECK(nvrtcGetLoweredName(program, name.c_str(), &loweredName), [&]{
        JST_ERROR("[CUDA] Can't get lowered name: {}", err);
    });
    
    // Create module.

    JST_CUDA_CHECK(cuModuleLoadData(&kernel.module, ptx.data()), [&]{
        JST_ERROR("[CUDA] Can't load module: {}", err);
    });

    // Get function.

    JST_CUDA_CHECK(cuModuleGetFunction(&kernel.function, kernel.module, loweredName), [&]{
        JST_ERROR("[CUDA] Can't get function: {}", err);
    });

    // Destroy program.

    JST_NVRTC_CHECK(nvrtcDestroyProgram(&program), [&]{
        JST_ERROR("[CUDA] Can't destroy program: {}", err);
    });

    return Result::SUCCESS;
}

Result CUDA::destroyKernel(const std::string& name) {
    auto& kernel = pimpl->kernels[name];

    JST_CUDA_CHECK(cuModuleUnload(kernel.module), [&]{
        JST_ERROR("[CUDA] Can't unload module: {}", err);
    });

    pimpl->kernels.erase(name);

    return Result::SUCCESS;
}

Result CUDA::launchKernel(const std::string& name, 
                          const std::vector<U64>& grid,
                          const std::vector<U64>& block,
                          void** arguments) {
    const auto& kernel = pimpl->kernels.at(name);

    JST_CUDA_CHECK(cuLaunchKernel(kernel.function, 
                                  grid[0], grid[1], grid[2], 
                                  block[0], block[1], block[2],
                                  0, _stream, arguments, 0), [&]{
        JST_ERROR("[CUDA] Can't launch kernel: {}", err);
    });

    return Result::SUCCESS;
}

}  // namespace Jetstream
