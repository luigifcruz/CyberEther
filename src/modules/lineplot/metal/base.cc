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
struct Lineplot<D, T>::Impl {
    struct MetalConstants {
        U16 batchSize;
        U16 gridSize;
    };

    MTL::ComputePipelineState* state;
    Tensor<Device::Metal, U8> constants;
};

template<Device D, typename T>
Lineplot<D, T>::Lineplot() {
    pimpl = std::make_unique<Impl>();
}

template<Device D, typename T>
Lineplot<D, T>::~Lineplot() {
    pimpl.reset();
}

template<Device D, typename T>
Result Lineplot<D, T>::createCompute(const Context& ctx) {
    JST_TRACE("Create Multiply compute core using Metal backend.");

    JST_CHECK(Metal::CompileKernel(shadersSrc, "lineplot", &pimpl->state));
    auto* constants = Metal::CreateConstants<Impl::Constants>(assets);
    constants->batchSize = numberOfBatches;
    constants->gridSize = numberOfElements;

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Lineplot<D, T>::compute(const Context& ctx) {
    auto cmdEncoder = ctx.metal->commandBuffer()->computeCommandEncoder();
    cmdEncoder->setComputePipelineState(pimpl->state);
    cmdEncoder->setBuffer(pimpl->constants.data(), 0, 0);
    cmdEncoder->setBuffer(input.buffer.data(), 0, 1);
    cmdEncoder->setBuffer(MapOn<Device::Metal>(plot).data(), 0, 2);
    cmdEncoder->dispatchThreads(MTL::Size(numberOfElements, 1, 1),
                                MTL::Size(pimpl->state->maxTotalThreadsPerThreadgroup(), 1, 1));
    cmdEncoder->endEncoding();

    return Result::SUCCESS;
}

JST_LINEPLOT_METAL(JST_INSTANTIATION)
JST_LINEPLOT_METAL(JST_BENCHMARK)

}  // namespace Jetstream
