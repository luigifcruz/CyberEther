#include "../generic.cc"

namespace Jetstream {

// TODO: Improve performance.

static const char shadersSrc[] = R"""(
    #include <metal_stdlib>

    using namespace metal;

    struct Constants {
        ushort batchSize;
        ushort gridSize;
        float normalizationFactor;
        size_t average;
    };

    // TODO: This can be ported to use shared memory and other tricks.
    //       Good enough for now.
    kernel void lineplot(constant Constants& constants [[ buffer(0) ]],
                         constant const float *input [[ buffer(1) ]],
                         device float2 *bins [[ buffer(2) ]],
                         uint id[[ thread_position_in_grid ]]) {
        if (id >= constants.gridSize) {
            return;
        }

        // Compute average amplitude within a batch.
        float amplitude = 0.0f;
        for (uint i = 0; i < constants.batchSize; ++i) {
            amplitude += input[id + (i * constants.gridSize)];
        }
        amplitude = (amplitude * constants.normalizationFactor) - 1.0f;

        // Calculate moving average.
        float average = bins[id].y;
        average -= average / constants.average;
        average += amplitude / constants.average;

        // Store result.
        bins[id].x = id * 2.0f / (constants.gridSize - 1) - 1.0f;
        bins[id].y = average;
    }
)""";

template<Device D, typename T>
struct Lineplot<D, T>::Impl {
    struct Constants {
        U16 batchSize;
        U16 gridSize;
        F32 normalizationFactor;
        U64 average;
    };

    MTL::ComputePipelineState* lineplotState;
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
Result Lineplot<D, T>::createCompute(const Context&) {
    JST_TRACE("Create Multiply compute core using Metal backend.");

    // Compile shaders.

    JST_CHECK(Metal::CompileKernel(shadersSrc, "lineplot", &pimpl->lineplotState));

    // Create constants buffer.

    Metal::CreateConstants<typename Impl::Constants>(*pimpl);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Lineplot<D, T>::compute(const Context& ctx) {
    auto* constants = Metal::Constants<typename Impl::Constants>(*pimpl);
    constants->batchSize = numberOfBatches;
    constants->gridSize = numberOfElements;
    constants->normalizationFactor = normalizationFactor;
    constants->average = config.averaging;

    {
        auto cmdEncoder = ctx.metal->commandBuffer()->computeCommandEncoder();
        cmdEncoder->setComputePipelineState(pimpl->lineplotState);
        cmdEncoder->setBuffer(pimpl->constants.data(), 0, 0);
        cmdEncoder->setBuffer(input.buffer.data(), 0, 1);
        cmdEncoder->setBuffer(signalPoints.data(), 0, 2);
        cmdEncoder->dispatchThreads(MTL::Size(numberOfElements, 1, 1),
                                    MTL::Size(pimpl->lineplotState->maxTotalThreadsPerThreadgroup(), 1, 1));
        cmdEncoder->endEncoding();
    }

    updateSignalPointsFlag = true;

    return Result::SUCCESS;
}

JST_LINEPLOT_METAL(JST_INSTANTIATION)
JST_LINEPLOT_METAL(JST_BENCHMARK)

}  // namespace Jetstream
