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

    kernel void scale(constant Constants& constants [[ buffer(0) ]],
                      constant const float *input [[ buffer(1) ]],
                      device float *output [[ buffer(2) ]],
                      uint id[[ thread_position_in_grid ]]) {
        // TODO: Can cache constants.max - constants.min.
        output[id] = (input[id] - constants.min) / (constants.max - constants.min);
    }
)""";

template<Device D, typename T>
Result Scale<D, T>::createCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Create Scale compute core using CPU backend.");

    auto& assets = metal;

    JST_CHECK(Metal::CompileKernel(shadersSrc, "scale", &assets.state));
    Metal::CreateConstants<MetalConstants>(assets);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Scale<D, T>::compute(const RuntimeMetadata& meta) {
    auto& assets = metal;
    auto& runtime = meta.metal;

    auto* constants = Metal::Constants<MetalConstants>(assets);
    constants->min = config.range.min;
    constants->max = config.range.max;

    auto cmdEncoder = runtime.commandBuffer->computeCommandEncoder();
    cmdEncoder->setComputePipelineState(metal.state);
    cmdEncoder->setBuffer(metal.constants.data(), 0, 0);
    cmdEncoder->setBuffer(input.buffer.data(), 0, 1);
    cmdEncoder->setBuffer(output.buffer.data(), 0, 2);
    cmdEncoder->dispatchThreads(MTL::Size(output.buffer.size(), 1, 1), 
                                MTL::Size(metal.state->maxTotalThreadsPerThreadgroup(), 1, 1));
    cmdEncoder->endEncoding();

    return Result::SUCCESS;
}

template class Scale<Device::Metal, F32>;
    
}  // namespace Jetstream
