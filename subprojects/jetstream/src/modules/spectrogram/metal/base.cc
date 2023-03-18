#include "../generic.cc"

namespace Jetstream {

static const char shadersSrc[] = R"""(
        #include <metal_stdlib>

        using namespace metal;

        kernel void multiply(
            const device float *input [[ buffer(0) ]],
            device float *bins [[ buffer(1) ]],
            uint id[[ thread_position_in_grid ]])
        {
            bins[id] *= 0.999;
        }
    )""";

template<Device D, typename T>
const Result Spectrogram<D, T>::createCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Create Spectrogram compute core using Metal backend.");

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

    frequencyBins = Vector<Device::Metal, F32>({input.buffer.size() * config.viewSize.height});

    return Result::SUCCESS;
}

template<Device D, typename T>
const Result Spectrogram<D, T>::compute(const RuntimeMetadata& meta) {
    // for (U64 x = 0; x < input.buffer.size() * config.viewSize.height; x++) {
    //     frequencyBins[x] *= 0.999; 
    // }
    //
    // for (U64 x = 0; x < input.buffer.size(); x++) {
    //     U16 index = input.buffer[x] * config.viewSize.height;
    //
    //     if (index < config.viewSize.height && index > 0) {
    //         frequencyBins[x + (index * input.buffer.size())] += 0.02; 
    //     }
    // }

    auto cmdEncoder = meta.metal.commandBuffer->computeCommandEncoder();
    cmdEncoder->setComputePipelineState(metal.state);
    cmdEncoder->setBuffer(input.buffer, 0, 0);
    cmdEncoder->setBuffer(frequencyBins, 0, 1);
    cmdEncoder->dispatchThreads(
            MTL::Size(input.buffer.size(), 1, 1),
            MTL::Size(metal.state->maxTotalThreadsPerThreadgroup(), 1, 1)
        );
    cmdEncoder->endEncoding();
    // cmdEncoder->release();

    auto blitEncoder = meta.metal.commandBuffer->blitCommandEncoder();
    blitEncoder->synchronizeResource(frequencyBins);
    blitEncoder->endEncoding();
    // blitEncoder->release();   

    return Result::SUCCESS;
}

// template class Spectrogram<Device::CPU, F64>;
template class Spectrogram<Device::Metal, F32>;

}  // namespace Jetstream
