#include "../generic.cc"

#include "jetstream/memory2/helpers.hh"

namespace Jetstream {

static const char shadersSrc[] = R"""(
    #include <metal_stdlib>

    using namespace metal;

    struct Constants {
        float constantReal;
        float constantImage;
    };

    kernel void multiply_complex(constant Constants& constants [[ buffer(0) ]],
                                 constant const float *factor [[ buffer(1) ]],
                                 device float *product [[ buffer(2) ]],
                                 uint id[[ thread_position_in_grid ]]) {
        const uint index = id * 2;
        product[index + 0] = (factor[index + 0] * constants.constantReal) -
                             (factor[index + 1] * constants.constantImage);
        product[index + 1] = (factor[index + 0] * constants.constantImage) +
                             (factor[index + 1] * constants.constantReal);
    }

    kernel void multiply(constant Constants& constants [[ buffer(0) ]],
                         constant const float *factor [[ buffer(1) ]],
                         device float *product [[ buffer(2) ]],
                         uint id[[ thread_position_in_grid ]]) {
        product[id] = factor[id + 0] * constants.constantReal;
    }
)""";

template<Device D, typename T>
struct MultiplyConstant<D, T>::Impl {
    struct MetalConstants {
        F32 constantReal;
        F32 constantImage;
    };

    struct {
        MTL::ComputePipelineState* state;
        mem2::Tensor constants;
    } metal;
};

template<Device D, typename T>
MultiplyConstant<D, T>::MultiplyConstant() {
    impl = std::make_unique<Impl>();
}

template<Device D, typename T>
MultiplyConstant<D, T>::~MultiplyConstant() {
    impl.reset();
}

template<Device D, typename T>
Result MultiplyConstant<D, T>::createCompute(const Context&) {
    JST_TRACE("Create Multiply Constant compute core using Metal backend.");

    auto* constants = Metal::CreateConstants<typename Impl::MetalConstants>(impl->metal);
    if constexpr (IsComplex<T>::value) {
        JST_CHECK(Metal::CompileKernel(shadersSrc, "multiply_complex", &impl->metal.state));
        constants->constantReal = config.constant.real();
        constants->constantImage = config.constant.imag();
    } else {
        JST_CHECK(Metal::CompileKernel(shadersSrc, "multiply", &impl->metal.state));
        constants->constantReal = config.constant;
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result MultiplyConstant<D, T>::compute(const Context& ctx) {
    auto cmdEncoder = ctx.metal->commandBuffer()->computeCommandEncoder();
    cmdEncoder->setComputePipelineState(impl->metal.state);
    cmdEncoder->setBuffer(impl->metal.constants.data(), 0, 0);
    cmdEncoder->setBuffer(input.factor.data(), 0, 1);
    cmdEncoder->setBuffer(output.product.data(), 0, 2);
    cmdEncoder->dispatchThreads(MTL::Size(output.product.size(), 1, 1),
                                MTL::Size(impl->metal.state->maxTotalThreadsPerThreadgroup(), 1, 1));
    cmdEncoder->endEncoding();

    return Result::SUCCESS;
}

JST_MULTIPLY_CONSTANT_METAL(JST_INSTANTIATION)
JST_MULTIPLY_CONSTANT_METAL(JST_BENCHMARK)

}  // namespace Jetstream
