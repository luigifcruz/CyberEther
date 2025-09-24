#include "../generic.cc"

#include "jetstream/memory2/helpers.hh"

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
struct Spectrogram<D, T>::Impl {
    struct Constants {
        U32 width;
        U32 height;
        F32 decayFactor;
        U32 batchSize;
    };

    MTL::ComputePipelineState* stateDecay;
    MTL::ComputePipelineState* stateActivate;
    mem2::Tensor constants;
};

template<Device D, typename T>
Spectrogram<D, T>::Spectrogram() {
    pimpl = std::make_unique<Impl>();
    gimpl = std::make_unique<GImpl>();
}

template<Device D, typename T>
Spectrogram<D, T>::~Spectrogram() {
    pimpl.reset();
    gimpl.reset();
}

template<Device D, typename T>
Result Spectrogram<D, T>::createCompute(const Context&) {
    JST_TRACE("Create Spectrogram compute core using Metal backend.");

    JST_CHECK(Metal::CompileKernel(shadersSrc, "decay", &pimpl->stateDecay));
    JST_CHECK(Metal::CompileKernel(shadersSrc, "activate", &pimpl->stateActivate));

    auto* constants = Metal::CreateConstants<typename Impl::Constants>(*pimpl);
    constants->width = gimpl->numberOfElements;
    constants->height = config.height;
    constants->decayFactor = gimpl->decayFactor;
    constants->batchSize = gimpl->numberOfBatches;

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Spectrogram<D, T>::compute(const Context& ctx) {
    {
        auto cmdEncoder = ctx.metal->commandBuffer()->computeCommandEncoder();
        cmdEncoder->setComputePipelineState(pimpl->stateDecay);
        cmdEncoder->setBuffer(pimpl->constants.data(), 0, 0);
        cmdEncoder->setBuffer(gimpl->frequencyBins.data(), 0, 1);

        auto w = pimpl->stateDecay->threadExecutionWidth();
        auto h = pimpl->stateDecay->maxTotalThreadsPerThreadgroup() / w;
        auto threadsPerThreadgroup = MTL::Size(w, h, 1);
        auto threadsPerGrid = MTL::Size(gimpl->numberOfElements, config.height, 1);
        cmdEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);

        cmdEncoder->endEncoding();
    }

    {
        auto cmdEncoder = ctx.metal->commandBuffer()->computeCommandEncoder();
        cmdEncoder->setComputePipelineState(pimpl->stateActivate);
        cmdEncoder->setBuffer(pimpl->constants.data(), 0, 0);
        cmdEncoder->setBuffer(input.buffer.data(), 0, 1);
        cmdEncoder->setBuffer(gimpl->frequencyBins.data(), 0, 2);

        auto w = pimpl->stateDecay->threadExecutionWidth();
        auto h = pimpl->stateDecay->maxTotalThreadsPerThreadgroup() / w;
        auto threadsPerThreadgroup = MTL::Size(w, h, 1);
        auto threadsPerGrid = MTL::Size(gimpl->numberOfElements, gimpl->numberOfBatches, 1);
        cmdEncoder->dispatchThreads(threadsPerGrid, threadsPerThreadgroup);

        cmdEncoder->endEncoding();
    }

    return Result::SUCCESS;
}

JST_SPECTROGRAM_METAL(JST_INSTANTIATION)
JST_SPECTROGRAM_METAL(JST_BENCHMARK)

}  // namespace Jetstream
