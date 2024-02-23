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
        float2 thickness;
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

        // Store result.
        bins[id].x = id * 2.0f / (constants.gridSize - 1) - 1.0f;
        bins[id].y = amplitude;
    }

    inline float2 ComputePerpendicular(float2 d, float2 thickness) {
        // Compute length
        const float length = sqrt(d.x * d.x + d.y * d.y);

        // Normalize
        d.x /= length;
        d.y /= length;

        // Return perperdicular (normalized)
        return {-d.y * thickness.x, d.x * thickness.y};
    }

    kernel void thickness(constant Constants& constants [[ buffer(0) ]],
                          constant const float2 *input [[ buffer(1) ]],
                          device float *output [[ buffer(2) ]],
                          uint id[[ thread_position_in_grid ]]) {
        if (id >= constants.gridSize - 1) {
            return;
        }

        const float2 p1 = input[id + 0];
        const float2 p2 = input[id + 1];

        const float2 d = {p2.x - p1.x, p2.y - p1.y};
        const float2 perp = ComputePerpendicular(d, constants.thickness);

        const size_t idx = id * 4 * 3;

        // Upper left
        output[idx] = p1.x + perp.x;
        output[idx + 1] = p1.y + perp.y;
        output[idx + 2] = 0.0f;

        // Lower left
        output[idx + 3] = p1.x - perp.x;
        output[idx + 4] = p1.y - perp.y;
        output[idx + 5] = 0.0f;

        // Upper right
        output[idx + 6] = p2.x + perp.x;
        output[idx + 7] = p2.y + perp.y;
        output[idx + 8] = 0.0f;

        // Lower right
        output[idx + 9] = p2.x - perp.x;
        output[idx + 10] = p2.y - perp.y;
        output[idx + 11] = 0.0f;
    }
)""";

template<Device D, typename T>
struct Lineplot<D, T>::Impl {
    struct Constants {
        U16 batchSize;
        U16 gridSize;
        F32 normalizationFactor;
        U64 average;
        std::pair<F32, F32> thickness;
    };

    MTL::ComputePipelineState* lineplotState;
    MTL::ComputePipelineState* thicklineState;
    Tensor<Device::Metal, U8> constants;
    Tensor<Device::Metal, T> intermediate;
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
    JST_CHECK(Metal::CompileKernel(shadersSrc, "thickness", &pimpl->thicklineState));

    // Create constants buffer.

    Metal::CreateConstants<typename Impl::Constants>(*pimpl);

    // Allocate intermediate buffer.

    pimpl->intermediate = Tensor<Device::Metal, T>({numberOfElements, 2});

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Lineplot<D, T>::compute(const Context& ctx) {
    auto* constants = Metal::Constants<typename Impl::Constants>(*pimpl);
    constants->batchSize = numberOfBatches;
    constants->gridSize = numberOfElements;
    constants->normalizationFactor = normalizationFactor;
    constants->average = config.averaging;
    constants->thickness = thickness;

    {
        auto cmdEncoder = ctx.metal->commandBuffer()->computeCommandEncoder();
        cmdEncoder->setComputePipelineState(pimpl->lineplotState);
        cmdEncoder->setBuffer(pimpl->constants.data(), 0, 0);
        cmdEncoder->setBuffer(input.buffer.data(), 0, 1);
        cmdEncoder->setBuffer(pimpl->intermediate.data(), 0, 2);
        cmdEncoder->dispatchThreads(MTL::Size(numberOfElements, 1, 1),
                                    MTL::Size(pimpl->lineplotState->maxTotalThreadsPerThreadgroup(), 1, 1));
        cmdEncoder->endEncoding();
    }

    {
        auto cmdEncoder = ctx.metal->commandBuffer()->computeCommandEncoder();
        cmdEncoder->setComputePipelineState(pimpl->thicklineState);
        cmdEncoder->setBuffer(pimpl->constants.data(), 0, 0);
        cmdEncoder->setBuffer(pimpl->intermediate.data(), 0, 1);
        cmdEncoder->setBuffer(MapOn<Device::Metal>(signal).data(), 0, 2);
        cmdEncoder->dispatchThreads(MTL::Size(numberOfElements, 1, 1),
                                    MTL::Size(pimpl->thicklineState->maxTotalThreadsPerThreadgroup(), 1, 1));
        cmdEncoder->endEncoding();
    }

    updateSignalVerticesFlag = true;

    return Result::SUCCESS;
}

JST_LINEPLOT_METAL(JST_INSTANTIATION)
JST_LINEPLOT_METAL(JST_BENCHMARK)

}  // namespace Jetstream
