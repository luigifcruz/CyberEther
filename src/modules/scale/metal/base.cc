#include "../generic.cc"

namespace Jetstream {

static const char shadersSrc[] = R"""(
    #include <metal_stdlib>
    #include <metal_math>

    using namespace metal;

    struct Constants {
        float scaling;
        float offset;
    };

    kernel void scale(constant Constants& constants [[ buffer(0) ]],
                      constant const float *input [[ buffer(1) ]],
                      device float *output [[ buffer(2) ]],
                      uint id[[ thread_position_in_grid ]]) {
        output[id] = input[id] * constants.scaling + constants.offset;
    }
)""";

template<Device D, typename T>
struct Scale<D, T>::Impl {
    struct Constants {
        F32 min;
        F32 max;
    };

    MTL::ComputePipelineState* state;
    Tensor<Device::Metal, U8> constants;
};

template<Device D, typename T>
Scale<D, T>::Scale() {
    pimpl = std::make_unique<Impl>();
}

template<Device D, typename T>
Scale<D, T>::~Scale() {
    pimpl.reset();
}

template<Device D, typename T>
Result Scale<D, T>::createCompute(const Context& ctx) {
    JST_TRACE("Create Scale compute core using CPU backend.");

    JST_CHECK(Metal::CompileKernel(shadersSrc, "scale", &pimpl->state));
    Metal::CreateConstants<typename Impl::Constants>(*pimpl);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Scale<D, T>::compute(const Context& ctx) {
    auto* constants = Metal::Constants<typename Impl::Constants>(*pimpl);
    constants->min = scalingCoeff;
    constants->max = offsetCoeff;

    auto cmdEncoder = ctx.metal->commandBuffer()->computeCommandEncoder();
    cmdEncoder->setComputePipelineState(pimpl->state);
    cmdEncoder->setBuffer(pimpl->constants.data(), 0, 0);
    cmdEncoder->setBuffer(input.buffer.data(), 0, 1);
    cmdEncoder->setBuffer(output.buffer.data(), 0, 2);
    cmdEncoder->dispatchThreads(MTL::Size(output.buffer.size(), 1, 1), 
                                MTL::Size(pimpl->state->maxTotalThreadsPerThreadgroup(), 1, 1));
    cmdEncoder->endEncoding();

    return Result::SUCCESS;
}

JST_SCALE_METAL(JST_INSTANTIATION)
JST_SCALE_METAL(JST_BENCHMARK)

}  // namespace Jetstream
