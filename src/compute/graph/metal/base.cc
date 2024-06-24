#include "jetstream/compute/graph/metal.hh"

namespace Jetstream {

Metal::Metal() {
    JST_DEBUG("Creating new Metal compute graph.");
    context = std::make_shared<Compute::Context>();
    context->metal = this;
}

Metal::~Metal() {
    context.reset();
}

Result Metal::create() {
    // TODO: Check if a inner pool is necessary.

    const auto& device = Backend::State<Device::Metal>()->getDevice();
    _commandQueue = device->newCommandQueue();

    for (const auto& computeUnit : computeUnits) {
        JST_CHECK(computeUnit.block->createCompute(*context));
    }

    return Result::SUCCESS;
}

Result Metal::computeReady() {
    for (const auto& computeUnit : computeUnits) {
        JST_CHECK(computeUnit.block->computeReady());
    }
    return Result::SUCCESS;
}

Result Metal::compute(std::unordered_set<U64>& yielded) {
    innerPool = NS::AutoreleasePool::alloc()->init();

    _commandBuffer = _commandQueue->commandBuffer();

    for (const auto& computeUnit : computeUnits) {
        if (Graph::ShouldYield(yielded, computeUnit.inputSet)) {
            Graph::YieldCompute(yielded, computeUnit.outputSet);
            continue;
        }

        const auto& res = computeUnit.block->compute(*context);

        if (res == Result::SUCCESS) {
            continue;
        }

        if (res == Result::YIELD) {
            Graph::YieldCompute(yielded, computeUnit.outputSet);
            continue;
        }

        JST_CHECK(res);
    }

    _commandBuffer->commit();
    _commandBuffer->waitUntilCompleted();

    innerPool->release();

    return Result::SUCCESS;
}

Result Metal::destroy() {
    for (const auto& computeUnit : computeUnits) {
        JST_CHECK(computeUnit.block->destroyCompute(*context));
    }
    computeUnits.clear();
    
    // TODO: Check if a inner pool is necessary.

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
        JST_ERROR("Error while compiling kernel library:\n{}", err->description()->utf8String());
        return Result::ERROR;
    }

    auto functionName = NS::String::string(methodName, NS::UTF8StringEncoding);
    auto kernel = library->newFunction(functionName);
    if (!kernel) {
        JST_ERROR("Error while creating Metal function.");
        return Result::ERROR;
    }

    if (*pipelineState = device->newComputePipelineState(kernel, MTL::PipelineOptionNone, 
                                                        nullptr, nullptr); !*pipelineState) {
        JST_ERROR("Error while creating Metal pipeline state.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream
