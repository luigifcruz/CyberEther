#include "../generic.cc"

namespace Jetstream {

// TODO: Implement thickline generation for Metal backend.

static const char shadersSrc[] = R"""(
    #include <metal_stdlib>

    using namespace metal;

    struct Constants {
        ushort batchSize;
        ushort gridSize;
        float normalizationFactor;
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
        bins[plot_idx] = (sum * constants.normalizationFactor) - 1.0f;
    }
)""";

template<Device D, typename T>
struct Lineplot<D, T>::Impl {
    struct Constants {
        U16 batchSize;
        U16 gridSize;
        F32 normalizationFactor;
    };

    MTL::ComputePipelineState* state;
    Tensor<Device::Metal, U8> constants;
};

template<Device D, typename T>
Lineplot<D, T>::Lineplot() {
    pimpl = std::make_unique<Impl>();
    gimpl = std::make_unique<GImpl>();
}

template<Device D, typename T>
Lineplot<D, T>::~Lineplot() {
    pimpl.reset();
    gimpl.reset();
}

template<Device D, typename T>
Result Lineplot<D, T>::createCompute(const Context& ctx) {
    JST_TRACE("Create Multiply compute core using Metal backend.");

    JST_CHECK(Metal::CompileKernel(shadersSrc, "lineplot", &pimpl->state));
    auto* constants = Metal::CreateConstants<typename Impl::Constants>(*pimpl);
    constants->batchSize = numberOfBatches;
    constants->gridSize = numberOfElements;
    constants->normalizationFactor = normalizationFactor;

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Lineplot<D, T>::compute(const Context& ctx) {
    auto cmdEncoder = ctx.metal->commandBuffer()->computeCommandEncoder();
    cmdEncoder->setComputePipelineState(pimpl->state);
    cmdEncoder->setBuffer(pimpl->constants.data(), 0, 0);
    cmdEncoder->setBuffer(input.buffer.data(), 0, 1);
    cmdEncoder->setBuffer(MapOn<Device::Metal>(signal).data(), 0, 2);
    cmdEncoder->dispatchThreads(MTL::Size(numberOfElements, 1, 1),
                                MTL::Size(pimpl->state->maxTotalThreadsPerThreadgroup(), 1, 1));
    cmdEncoder->endEncoding();

    return Result::SUCCESS;
}

JST_LINEPLOT_METAL(JST_INSTANTIATION)
JST_LINEPLOT_METAL(JST_BENCHMARK)

}  // namespace Jetstream
