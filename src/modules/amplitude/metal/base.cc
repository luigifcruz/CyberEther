#include "../generic.cc"

namespace Jetstream {

static const char shadersSrcComplex[] = R"""(
    #include <metal_stdlib>
    #include <metal_math>

    using namespace metal;

    struct Constants {
        float scalingCoeff;
    };

    kernel void amplitude(constant Constants& constants [[ buffer(0) ]],
                          constant const float2 *input [[ buffer(1) ]],
                          device float *output [[ buffer(2) ]],
                          uint id[[ thread_position_in_grid ]]) {
        float2 number = input[id];
        float real = number.x;
        float imag = number.y;
        float pwr = sqrt((real * real) + (imag * imag));
        output[id] = 20.0 * log10(pwr) + constants.scalingCoeff;
    }
)""";

static const char shadersSrcReal[] = R"""(
    #include <metal_stdlib>
    #include <metal_math>

    using namespace metal;

    struct Constants {
        float scalingCoeff;
    };

    kernel void amplitude(constant Constants& constants [[ buffer(0) ]],
                          constant const float *input [[ buffer(1) ]],
                          device float *output [[ buffer(2) ]],
                          uint id[[ thread_position_in_grid ]]) {
        float pwr = fabs(input[id]);
        output[id] = 20.0 * log10(pwr) + constants.scalingCoeff;
    }
)""";

template<Device D, typename IT, typename OT>
struct Amplitude<D, IT, OT>::Impl {
    struct Constants {
        F32 scalingCoeff;
    };

    MTL::ComputePipelineState* state;
    Tensor<Device::Metal, U8> constants;
};

template<Device D, typename IT, typename OT>
Amplitude<D, IT, OT>::Amplitude() {
    pimpl = std::make_unique<Impl>();
}

template<Device D, typename IT, typename OT>
Amplitude<D, IT, OT>::~Amplitude() {
    pimpl.reset();
}

template<Device D, typename IT, typename OT>
Result Amplitude<D, IT, OT>::createCompute(const Context& ctx) {
    JST_TRACE("Create Amplitude compute core using Metal backend.");

    if constexpr (std::is_same_v<IT, CF32> && std::is_same_v<OT, F32>) {
        return JST_CHECK(Metal::CompileKernel(shadersSrcComplex, "amplitude", &pimpl->state));
    } else if constexpr (std::is_same_v<IT, F32> && std::is_same_v<OT, F32>) {
        return JST_CHECK(Metal::CompileKernel(shadersSrcReal, "amplitude", &pimpl->state));
    }

    auto* constants = Metal::CreateConstants<typename Impl::Constants>(*pimpl);
    constants->scalingCoeff = scalingCoeff;

    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
Result Amplitude<D, IT, OT>::compute(const Context& ctx) {
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

JST_AMPLITUDE_METAL(JST_INSTANTIATION)
JST_AMPLITUDE_METAL(JST_BENCHMARK)
    
}  // namespace Jetstream
