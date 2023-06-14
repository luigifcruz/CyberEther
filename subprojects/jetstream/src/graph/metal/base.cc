#include "jetstream/graph/metal.hh"

namespace Jetstream {

Metal::Metal() {
    JST_DEBUG("Creating new Metal compute graph.");
    metadata = std::make_shared<RuntimeMetadata>();

    metadata->metal.commandBuffer = nullptr;

    auto device = Backend::State<Device::Metal>()->getDevice();
    metadata->metal.commandQueue = device->newCommandQueue();
}

Result Metal::createCompute() {
    outerPool = NS::AutoreleasePool::alloc()->init();

    for (const auto& block : blocks) {
        JST_CHECK(block->createCompute(*metadata));
    }

    return Result::SUCCESS;
}

Result Metal::compute() {
    Result err = Result::SUCCESS;

    innerPool = NS::AutoreleasePool::alloc()->init();

    metadata->metal.commandBuffer = metadata->metal.commandQueue->commandBuffer();

    for (const auto& block : blocks) {
        if ((err = block->compute(*metadata)) != Result::SUCCESS) {
            if (err == Result::SKIP) {
                return Result::SUCCESS;
            }
            return err;
        }
    }
    
    // TODO: Add automatic blit synchronization of wired output buffers.

    metadata->metal.commandBuffer->commit();
    metadata->metal.commandBuffer->waitUntilCompleted();

    innerPool->release();

    return err;
}

Result Metal::destroyCompute() {
    for (const auto& block : blocks) {
        JST_CHECK(block->destroyCompute(*metadata));
    }

    outerPool->release();

    return Result::SUCCESS;
}

Result Metal::CompileKernel(const char* shaderSrc,
                            const char* methodName, 
                            MTL::ComputePipelineState** pipelineState) {
    auto device = Backend::State<Device::Metal>()->getDevice();

    MTL::CompileOptions* opts = MTL::CompileOptions::alloc()->init();
    opts->setFastMathEnabled(true);
    opts->setLanguageVersion(MTL::LanguageVersion3_0);

    NS::Error* err = nullptr;
    NS::String* source = NS::String::string(shaderSrc, NS::UTF8StringEncoding);
    auto library = device->newLibrary(source, opts, &err);
    if (!library) {
        JST_FATAL("Error while compiling kernel library:\n{}", err->description()->utf8String());
        return Result::ERROR;
    }

    auto functionName = NS::String::string(methodName, NS::UTF8StringEncoding);
    auto kernel = library->newFunction(functionName);
    if (!kernel) {
        JST_FATAL("Error while creating Metal function.");
        return Result::ERROR;
    }

    if (*pipelineState = device->newComputePipelineState(kernel, MTL::PipelineOptionNone, 
                                                        nullptr, nullptr); !*pipelineState) {
        JST_FATAL("Error while creating Metal pipeline state.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream
