#include "../generic.cc"

namespace Jetstream {

static const char shadersSrc[] = R"""(
    #include <metal_stdlib>
    #include <metal_math>

    using namespace metal;

    // ported from cuComplex
    float jsAbs(float2 x) {
        float a = x.x;
        float b = x.y;
        float v, w, t;
        a = abs(a);
        b = abs(b);
        if (a > b) {
            v = a;
            w = b; 
        } else {
            v = b;
            w = a;
        }
        t = w / v;
        t = 1.0f + t * t;
        t = v * sqrt(t);
        if ((v == 0.0f) || (v > 3.402823466e38f) || (w > 3.402823466e38f)) {
            t = v + w;
        }
        return t;
    }

    struct Constants {
        float scalingSize;
    };

    kernel void amplitude(constant Constants& constants [[ buffer(0) ]],
                          constant const float2 *input [[ buffer(1) ]],
                          device float *output [[ buffer(2) ]],
                          uint id[[ thread_position_in_grid ]]) {
        output[id] = 20.0 * log10(jsAbs(input[id]) / constants.scalingSize);
    }
)""";

template<Device D, typename IT, typename OT>
Result Amplitude<D, IT, OT>::createCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Create Amplitude compute core using Metal backend.");

    auto& assets = metal;

    JST_CHECK(Metal::CompileKernel(shadersSrc, "amplitude", &assets.state));
    auto* constants = Metal::CreateConstants<MetalConstants>(assets);
    constants->scalingSize = input.buffer.shape()[1];

    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
Result Amplitude<D, IT, OT>::compute(const RuntimeMetadata& meta) {
    auto& assets = metal;
    auto& runtime = meta.metal;

    auto cmdEncoder = runtime.commandBuffer->computeCommandEncoder();
    cmdEncoder->setComputePipelineState(assets.state);
    cmdEncoder->setBuffer(assets.constants, 0, 0);
    cmdEncoder->setBuffer(input.buffer, 0, 1);
    cmdEncoder->setBuffer(output.buffer, 0, 2);
    cmdEncoder->dispatchThreads(MTL::Size(output.buffer.size(), 1, 1),
                                MTL::Size(assets.state->maxTotalThreadsPerThreadgroup(), 1, 1));
    cmdEncoder->endEncoding();

    return Result::SUCCESS;
}

template class Amplitude<Device::Metal, CF32>;
    
}  // namespace Jetstream
