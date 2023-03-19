#include "../generic.cc"

namespace Jetstream {

static const char shadersSrc[] = R"""(
    #include <metal_stdlib>

    using namespace metal;

    struct Constants {
        uint64_t width;
        uint64_t height;
    };

    kernel void spectrogram(constant Constants& constants [[ buffer(0) ]],
                            const device float *input [[ buffer(1) ]],
                            device float *bins [[ buffer(2) ]],
                            uint2 gid[[ thread_position_in_grid ]]) {
        uint id = gid.y * constants.width + gid.x;

        bins[id] *= 0.999;

        if (gid.y == 0) {
            uint16_t dS = input[id] * constants.height;
            if (dS < constants.height && dS > 0) {
                bins[id + (dS * constants.width)] += 0.02;
            }
        }
    }
)""";

template<Device D, typename T>
const Result Spectrogram<D, T>::createCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Create Spectrogram compute core using Metal backend.");

    auto& assets = metal;

    JST_CHECK(Metal::CompileKernel(shadersSrc, "spectrogram", &assets.state));
    auto* constants = Metal::CreateConstants<MetalConstants>(assets);
    constants->width = input.buffer.size();
    constants->height = config.viewSize.height; 
    frequencyBins = Vector<Device::Metal, F32>({input.buffer.size() * config.viewSize.height});

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

    auto cmdEncoder = runtime.commandBuffer->computeCommandEncoder();
    cmdEncoder->setComputePipelineState(assets.state);
    cmdEncoder->setBuffer(assets.constants, 0, 0);
    cmdEncoder->setBuffer(input.buffer, 0, 1);
    cmdEncoder->setBuffer(frequencyBins, 0, 2);
    cmdEncoder->dispatchThreads(MTL::Size(input.buffer.size(), config.viewSize.height, 1),
                                MTL::Size(assets.state->maxTotalThreadsPerThreadgroup(), 1, 1));
    cmdEncoder->endEncoding();

    return Result::SUCCESS;
}

template class Spectrogram<Device::Metal, F32>;

}  // namespace Jetstream
