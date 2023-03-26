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
                            device float *bins [[ buffer(2) ]],
                            uint id[[ thread_position_in_grid ]]) {
        const ushort min = 0;
        const ushort max = constants.height;

        for (uint b = 0; b < constants.batchSize; b++) {
            const uint offset = b * constants.width;
            const ushort val = input[id + offset] * constants.height;

            if (val > min && val < max) {
                bins[id + (val * constants.width)] += 0.01;
            }
        }
    }
)""";

template<Device D, typename T>
const Result Spectrogram<D, T>::createCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Create Spectrogram compute core using Metal backend.");

    auto& assets = metal;

    JST_CHECK(Metal::CompileKernel(shadersSrc, "decay", &assets.stateDecay));
    JST_CHECK(Metal::CompileKernel(shadersSrc, "activate", &assets.stateActivate));

    frequencyBins = Vector<Device::Metal, F32, 2>({input.buffer.shape(1), config.viewSize.height});
    decayFactor = pow(0.999, input.buffer.shape(0));

    auto* constants = Metal::CreateConstants<MetalConstants>(assets);
    constants->width = input.buffer.shape(1);
    constants->height = config.viewSize.height; 
    constants->decayFactor = decayFactor;
    constants->batchSize = input.buffer.shape(0);

    return Result::SUCCESS;
}

template<Device D, typename T>
const Result Spectrogram<D, T>::viewSizeCallback() {
    auto& assets = metal;
    auto* constants = Metal::Constants<MetalConstants>(assets);
    constants->height = config.viewSize.height;

    return Result::SUCCESS;
}

template<Device D, typename T>
const Result Spectrogram<D, T>::compute(const RuntimeMetadata& meta) {
    auto& assets = metal;
    auto& runtime = meta.metal;

    {
        auto cmdEncoder = runtime.commandBuffer->computeCommandEncoder();
        cmdEncoder->setComputePipelineState(assets.stateDecay);
        cmdEncoder->setBuffer(assets.constants, 0, 0);
        cmdEncoder->setBuffer(frequencyBins, 0, 1);

        auto w = assets.stateDecay->threadExecutionWidth();
        auto h = assets.stateDecay->maxTotalThreadsPerThreadgroup() / w;
        auto threadsPerThreadgroup = MTL::Size(w, h, 1);
        auto threadsPerGrid = MTL::Size(input.buffer.shape(1), config.viewSize.height, 1);
        cmdEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);

        cmdEncoder->endEncoding();
    }

    {
        auto cmdEncoder = runtime.commandBuffer->computeCommandEncoder();
        cmdEncoder->setComputePipelineState(assets.stateActivate);
        cmdEncoder->setBuffer(assets.constants, 0, 0);
        cmdEncoder->setBuffer(input.buffer, 0, 1);
        cmdEncoder->setBuffer(frequencyBins, 0, 2);

        auto w = assets.stateDecay->maxTotalThreadsPerThreadgroup();
        auto threadsPerThreadgroup = MTL::Size(w, 1, 1);
        auto threadsPerGrid = MTL::Size(input.buffer.shape(1), 1, 1);
        cmdEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);

        cmdEncoder->endEncoding();
    }

    return Result::SUCCESS;
}

template class Spectrogram<Device::Metal, F32>;

}  // namespace Jetstream
