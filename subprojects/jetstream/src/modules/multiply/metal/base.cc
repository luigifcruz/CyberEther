#include "../generic.cc"

namespace Jetstream {

static const char shadersSrc[] = R"""(
        #include <metal_stdlib>

        using namespace metal;

        kernel void multiply(
            const device float *factorA [[ buffer(0) ]],
            const device float *factorB [[ buffer(1) ]],
            device float *product [[ buffer(2) ]],
            uint id[[ thread_position_in_grid ]])
        {
            uint index = id * 2;
            product[index + 0] = (factorA[index + 0] * (factorB[index + 0])) - 
                                 (factorA[index + 1] * (factorB[index + 1]));
            product[index + 1] = (factorA[index + 0] * (factorB[index + 1])) + 
                                 (factorA[index + 1] * (factorB[index + 0]));
        }
    )""";

template<Device D, typename T>
const Result Multiply<D, T>::createCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Create Multiply compute core using Metal backend.");

    NS::Error* err = nullptr;
    auto device = Backend::State<Device::Metal>()->getDevice();
    MTL::CompileOptions* opts = MTL::CompileOptions::alloc();
    NS::String* source = NS::String::string(shadersSrc, NS::ASCIIStringEncoding);
    auto library = device->newLibrary(source, opts, &err);
    if (!library) {
        JST_FATAL("Library error:\n{}", err->description()->utf8String());
        return Result::ERROR;
    }
    auto kernel = library->newFunction(NS::String::string("multiply", NS::ASCIIStringEncoding));
    assert(kernel);
    metal.state = device->newComputePipelineState(kernel, MTL::PipelineOptionNone, nullptr, nullptr);
    assert(metal.state);

    return Result::SUCCESS;
}

template<Device D, typename T>
const Result Multiply<D, T>::compute(const RuntimeMetadata& meta) {
    auto cmdEncoder = meta.metal.commandBuffer->computeCommandEncoder();
    cmdEncoder->setComputePipelineState(metal.state);
    cmdEncoder->setBuffer(input.factorA.buffer(), 0, 0);
    cmdEncoder->setBuffer(input.factorB.buffer(), 0, 1);
    cmdEncoder->setBuffer(output.product.buffer(), 0, 2);
    cmdEncoder->dispatchThreads(
            MTL::Size(output.product.size(), 1, 1),
            MTL::Size(metal.state->maxTotalThreadsPerThreadgroup(), 1, 1)
        );
    cmdEncoder->endEncoding();
    // cmdEncoder->release();

    auto blitEncoder = meta.metal.commandBuffer->blitCommandEncoder();
    blitEncoder->synchronizeResource(output.product.buffer());
    blitEncoder->endEncoding();
    // blitEncoder->release();

    return Result::SUCCESS;
}

template class Multiply<Device::Metal, CF32>;
    
}  // namespace Jetstream
