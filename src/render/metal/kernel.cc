#include "jetstream/render/metal/buffer.hh"
#include "jetstream/render/metal/kernel.hh"

namespace Jetstream::Render {

using Implementation = KernelImp<Device::Metal>;

Implementation::KernelImp(const Config& config) : Kernel(config) {
    for (auto& buffer : config.buffers) {
        buffers.push_back(
            std::dynamic_pointer_cast<BufferImp<Device::Metal>>(buffer)
        );
    }
}

Result Implementation::create() {
    JST_DEBUG("Creating Metal kernel.");

    if (config.shaders.contains(Device::Metal) == 0) {
        JST_ERROR("[Metal] Module doesn't have necessary kernel.");       
        return Result::ERROR;
    }

    NS::Error* err = nullptr;
    const auto& shaders = config.shaders[Device::Metal];
    auto device = Backend::State<Device::Metal>()->getDevice();

    MTL::CompileOptions* opts = MTL::CompileOptions::alloc()->init();
    opts->setFastMathEnabled(true);
    opts->setLanguageVersion(MTL::LanguageVersion3_0);
    opts->setLibraryType(MTL::LibraryTypeExecutable);

    auto source = NS::String::alloc()->init((char*)shaders[0].data(), shaders[0].size(), NS::UTF8StringEncoding, false);
    auto library = device->newLibrary(source, opts, &err);

    if (!library) {
        JST_ERROR("Library error:\n{}", err->description()->utf8String());
        return Result::ERROR;
    }

    MTL::Function* func = library->newFunction(
        NS::String::string("main0", NS::UTF8StringEncoding)
    );
    JST_ASSERT(func);

    pipelineState = device->newComputePipelineState(func, &err);
    if (!pipelineState) {
        JST_ERROR("Failed to create pipeline state:\n{}", err->description()->utf8String());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    pipelineState->release();

    return Result::SUCCESS;
}

Result Implementation::encode(MTL::ComputeCommandEncoder* encoder) {
    // Set pipeline state.

    encoder->setComputePipelineState(pipelineState);

    // Attach compute buffers.

    for (U64 i = 0; i < buffers.size(); i++) {
        encoder->setBuffer(buffers[i]->getHandle(), 0, i);
    }

    // Dispatch threads.

    const auto& [x, y, z] = config.gridSize;

    // TODO: Implement 2D and 3D grid sizes.
    if (y != 1 || z != 1) {
        JST_ERROR("Only 1D grids are supported.");
        return Result::ERROR;
    }

    encoder->dispatchThreads(MTL::Size(x, y, z),
                             MTL::Size(pipelineState->maxTotalThreadsPerThreadgroup(), 1, 1));

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
