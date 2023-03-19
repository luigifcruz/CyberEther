#include "jetstream/graph/metal.hh"

namespace Jetstream {

Metal::Metal() {
    JST_DEBUG("Creating new Metal compute graph.");
    metadata = std::make_shared<RuntimeMetadata>();

    metadata->metal.commandBuffer = nullptr;

    auto device = Backend::State<Device::Metal>()->getDevice();
    metadata->metal.commandQueue = device->newCommandQueue();
}

const Result Metal::createCompute() {
    for (const auto& block : blocks) {
        JST_CHECK(block->createCompute(*metadata));
    }

    return Result::SUCCESS;
}

const Result Metal::compute() {
    Result err = Result::SUCCESS;

    loopPool = NS::AutoreleasePool::alloc()->init();

    metadata->metal.commandBuffer = metadata->metal.commandQueue->commandBuffer();

    for (const auto& block : blocks) {
        if ((err = block->compute(*metadata)) != Result::SUCCESS) {
            return err;
        }
    }
    
    // TODO: Add automatic blit synchronization of wired output buffers.

    metadata->metal.commandBuffer->commit();
    metadata->metal.commandBuffer->waitUntilCompleted();

    loopPool->release();

    return err;
}

const Result Metal::destroyCompute() {
    for (const auto& block : blocks) {
        JST_CHECK(block->destroyCompute(*metadata));
    }

    return Result::SUCCESS;
}

const Result Metal::CompileKernel(const char* shaderSrc,
                                  const char* methodName, 
                                  MTL::ComputePipelineState** pipelineState) {
    auto device = Backend::State<Device::Metal>()->getDevice();

    MTL::CompileOptions* opts = MTL::CompileOptions::alloc();
    opts->setFastMathEnabled(true);

    NS::Error* err = nullptr;
    NS::String* source = NS::String::string(shaderSrc, NS::ASCIIStringEncoding);
    auto library = device->newLibrary(source, opts, &err);
    if (!library) {
        JST_FATAL("Error while compiling kernel library:\n{}", err->description()->utf8String());
        return Result::ERROR;
    }

    auto functionName = NS::String::string(methodName, NS::ASCIIStringEncoding);
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
