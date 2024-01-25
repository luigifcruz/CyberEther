#include "../generic.cc"

namespace Jetstream {

static const char shadersSrc[] = R"""(
    #include <metal_stdlib>
    #include <metal_math>

    using namespace metal;

    struct Constants {
        float scalingCoeff;
    };

    kernel void amplitude(constant Constants& constants [[ buffer(0) ]],
                          constant const float2 *input [[ buffer(1) ]],
                          device float *output [[ buffer(2) ]],
                          uint id[[ thread_position_in_grid ]]) {
        float2 number = input[id];
        float real = number.x;
        float imag = number.y;
        float pwr = sqrt((real * real) + (imag * imag));
        output[id] = 20.0 * log10(pwr) + constants.scalingCoeff;
    }
)""";

template<Device D, typename IT, typename OT>
Result Amplitude<D, IT, OT>::createCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Create Amplitude compute core using Metal backend.");

    auto& assets = metal;

    JST_CHECK(Metal::CompileKernel(shadersSrc, "amplitude", &assets.state));
    auto* constants = Metal::CreateConstants<MetalConstants>(assets);
    constants->scalingCoeff = scalingCoeff;

    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
Result Amplitude<D, IT, OT>::compute(const RuntimeMetadata& meta) {
    auto& assets = metal;
    auto& runtime = meta.metal;

    auto cmdEncoder = runtime.commandBuffer->computeCommandEncoder();
    cmdEncoder->setComputePipelineState(assets.state);
    cmdEncoder->setBuffer(assets.constants.data(), 0, 0);
    cmdEncoder->setBuffer(input.buffer.data(), 0, 1);
    cmdEncoder->setBuffer(output.buffer.data(), 0, 2);
    cmdEncoder->dispatchThreads(MTL::Size(output.buffer.size(), 1, 1),
                                MTL::Size(assets.state->maxTotalThreadsPerThreadgroup(), 1, 1));
    cmdEncoder->endEncoding();

    return Result::SUCCESS;
}

JST_AMPLITUDE_METAL(JST_INSTANTIATION)
JST_AMPLITUDE_METAL(JST_BENCHMARK)
    
}  // namespace Jetstream
