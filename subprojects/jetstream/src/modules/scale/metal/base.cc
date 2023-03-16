#include "../generic.cc"

namespace Jetstream {

static const char shadersSrc[] = R"""(
        #include <metal_stdlib>
        #include <metal_math>

        using namespace metal;
    
        struct Constants {
            float min;
            float max;
        };

        kernel void scale(
            constant Constants& constants [[ buffer(0) ]],
            const device float *input [[ buffer(1) ]],
            device float *output [[ buffer(2) ]],
            uint id[[ thread_position_in_grid ]])
        {
            // TODO: Can cache constants.max - constants.min.
            output[id] = (input[id] - constants.min) / (constants.max - constants.min);
        }
    )""";

template<Device D, typename T>
const Result Scale<D, T>::createCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Create Scale compute core using CPU backend.");

    NS::Error* err = nullptr;
    auto device = Backend::State<Device::Metal>()->getDevice();
    MTL::CompileOptions* opts = MTL::CompileOptions::alloc();
    NS::String* source = NS::String::string(shadersSrc, NS::ASCIIStringEncoding);
    auto library = device->newLibrary(source, opts, &err);
    if (!library) {
        JST_FATAL("Library error:\n{}", err->description()->utf8String());
        return Result::ERROR;
    }
    auto kernel = library->newFunction(NS::String::string("scale", NS::ASCIIStringEncoding));
    assert(kernel);
    metal.state = device->newComputePipelineState(kernel, MTL::PipelineOptionNone, nullptr, nullptr);
    assert(metal.state);

    metal.constants = Vector<Device::CPU, U8>({sizeof(MetalConstants)});

    return Result::SUCCESS;
}

template<Device D, typename T>
const Result Scale<D, T>::compute(const RuntimeMetadata& meta) {
    auto [min, max] = this->config.range;
    (*reinterpret_cast<MetalConstants*>(metal.constants.data())).min = min;
    (*reinterpret_cast<MetalConstants*>(metal.constants.data())).max = max;

    auto cmdEncoder = meta.metal.commandBuffer->computeCommandEncoder();
    cmdEncoder->setComputePipelineState(metal.state);
    cmdEncoder->setBuffer(metal.constants.buffer(), 0, 0);
    cmdEncoder->setBuffer(input.buffer.buffer(), 0, 1);
    cmdEncoder->setBuffer(output.buffer.buffer(), 0, 2);
    cmdEncoder->dispatchThreads(
            MTL::Size(output.buffer.size(), 1, 1),
            MTL::Size(metal.state->maxTotalThreadsPerThreadgroup(), 1, 1)
        );
    cmdEncoder->endEncoding();
    // cmdEncoder->release();

    return Result::SUCCESS;
}

// TODO: Put this back.
// template class Scale<Device::CPU, F64>;
template class Scale<Device::Metal, F32>;
    
}  // namespace Jetstream
