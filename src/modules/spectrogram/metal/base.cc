#include "../generic.cc"

namespace Jetstream {

static const char shadersSrc[] = R"""(
    #include <metal_stdlib>
    #include <metal_compute>

    using namespace metal;

    struct Constants {
        uint width;
        uint height;
        float decayFactor;
        uint batchSize;
    };

    kernel void decay(constant Constants& constants [[ buffer(0) ]],
                      device float *bins [[ buffer(1) ]],
                      uint2 gid[[ thread_position_in_grid ]]) {
        const uint id = gid.y * constants.width + gid.x;
        bins[id] *= constants.decayFactor;
    }

    kernel void activate(constant Constants& constants [[ buffer(0) ]],
                            constant const float *input [[ buffer(1) ]],
                            device atomic_float *bins [[ buffer(2) ]],
                            uint2 gid[[ thread_position_in_grid ]]) {
        const ushort min = 0;
        const ushort max = constants.height;
        const uint offset = gid.y * constants.width;
        const ushort val = input[gid.x + offset] * constants.height;

        if (val > min && val < max) {
            atomic_fetch_add_explicit(&bins[gid.x + (val * constants.width)], 0.01f, memory_order_relaxed);
        }
    }
)""";

template<Device D, typename T>
Result Spectrogram<D, T>::createCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Create Spectrogram compute core using Metal backend.");

    auto& assets = metal;

    JST_CHECK(Metal::CompileKernel(shadersSrc, "decay", &assets.stateDecay));
    JST_CHECK(Metal::CompileKernel(shadersSrc, "activate", &assets.stateActivate));

    decayFactor = pow(0.999, numberOfBatches);

    auto* constants = Metal::CreateConstants<MetalConstants>(assets);
    constants->width = numberOfElements;
    constants->height = config.height; 
    constants->decayFactor = decayFactor;
    constants->batchSize = numberOfBatches;

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Spectrogram<D, T>::compute(const RuntimeMetadata& meta) {
    auto& assets = metal;
    auto& runtime = meta.metal;

    {
        auto cmdEncoder = runtime.commandBuffer->computeCommandEncoder();
        cmdEncoder->setComputePipelineState(assets.stateDecay);
        cmdEncoder->setBuffer(assets.constants.data(), 0, 0);
        cmdEncoder->setBuffer(frequencyBins.data(), 0, 1);

        auto w = assets.stateDecay->threadExecutionWidth();
        auto h = assets.stateDecay->maxTotalThreadsPerThreadgroup() / w;
        auto threadsPerThreadgroup = MTL::Size(w, h, 1);
        auto threadsPerGrid = MTL::Size(numberOfElements, config.height, 1);
        cmdEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);

        cmdEncoder->endEncoding();
    }

    {
        auto cmdEncoder = runtime.commandBuffer->computeCommandEncoder();
        cmdEncoder->setComputePipelineState(assets.stateActivate);
        cmdEncoder->setBuffer(assets.constants.data(), 0, 0);
        cmdEncoder->setBuffer(input.buffer.data(), 0, 1);
        cmdEncoder->setBuffer(frequencyBins.data(), 0, 2);

        auto w = assets.stateDecay->threadExecutionWidth();
        auto h = assets.stateDecay->maxTotalThreadsPerThreadgroup() / w;
        auto threadsPerThreadgroup = MTL::Size(w, h, 1);
        auto threadsPerGrid = MTL::Size(numberOfElements, numberOfBatches, 1);
        cmdEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);

        cmdEncoder->endEncoding();
    }

    return Result::SUCCESS;
}

JST_SPECTROGRAM_METAL(JST_INSTANTIATION)
JST_SPECTROGRAM_METAL(JST_BENCHMARK)

}  // namespace Jetstream
