#include "jetstream/modules/lineplot.hh"

#include "../generic.cc"

namespace Jetstream {

static const char shadersSrc[] = R"""(
    #include <metal_stdlib>

    using namespace metal;

    struct Constants {
        ushort batchSize;
        ushort gridSize;
    };

    // TODO: This can be ported to use shared memory and other tricks.
    //       Good enough for now.
    kernel void lineplot(constant Constants& constants [[ buffer(0) ]],
                         constant const float *input [[ buffer(1) ]],
                         device float *bins [[ buffer(2) ]],
                         uint id[[ thread_position_in_grid ]]) {
        float sum = 0.0f;
        for (uint i = 0; i < constants.batchSize; ++i) {
            sum += input[id + (i * constants.gridSize)];
        }

        const uint plot_idx = id * 3 + 1;
        bins[plot_idx] = (sum / (0.5f * constants.batchSize)) - 1.0f;
    }
)""";

template<Device D, typename T>
const Result Lineplot<D, T>::underlyingCreateCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Create Multiply compute core using Metal backend.");

    auto& assets = metal;

    JST_CHECK(Metal::CompileKernel(shadersSrc, "lineplot", &assets.state));
    auto* constants = Metal::CreateConstants<MetalConstants>(assets);
    constants->batchSize = this->input.buffer.shape(0);
    constants->gridSize = this->input.buffer.shape(1);

    return Result::SUCCESS;
}

template<Device D, typename T>
const Result Lineplot<D, T>::compute(const RuntimeMetadata& meta) {
    auto& assets = metal;
    auto& runtime = meta.metal;
    
    auto cmdEncoder = runtime.commandBuffer->computeCommandEncoder();
    cmdEncoder->setComputePipelineState(assets.state);
    cmdEncoder->setBuffer(assets.constants, 0, 0);
    cmdEncoder->setBuffer(input.buffer, 0, 1);
    cmdEncoder->setBuffer(plot, 0, 2);
    cmdEncoder->dispatchThreads(MTL::Size(input.buffer.shape(1), 1, 1),
                                MTL::Size(assets.state->maxTotalThreadsPerThreadgroup(), 1, 1));
    cmdEncoder->endEncoding();

    return Result::SUCCESS;
}

template class Lineplot<Device::Metal, F32>;
    
}  // namespace Jetstream
