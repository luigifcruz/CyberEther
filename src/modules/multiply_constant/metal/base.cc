#include "../generic.cc"

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
Result MultiplyConstant<D, T>::createCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Create Multiply Constant compute core using Metal backend.");

    auto& assets = metal;

    auto* constants = Metal::CreateConstants<MetalConstants>(assets);
    if constexpr (IsComplex<T>::value) {
        JST_CHECK(Metal::CompileKernel(shadersSrc, "multiply_complex", &assets.state));
        constants->constantReal = config.constant.real();
        constants->constantImage = config.constant.imag();
    } else {
        JST_CHECK(Metal::CompileKernel(shadersSrc, "multiply", &assets.state));
        constants->constantReal = config.constant;
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result MultiplyConstant<D, T>::compute(const RuntimeMetadata& meta) {
    auto& assets = metal;
    auto& runtime = meta.metal;

    auto cmdEncoder = runtime.commandBuffer->computeCommandEncoder();
    cmdEncoder->setComputePipelineState(assets.state);
    cmdEncoder->setBuffer(assets.constants, 0, 0);
    cmdEncoder->setBuffer(input.factor, 0, 1);
    cmdEncoder->setBuffer(output.product, 0, 2);
    cmdEncoder->dispatchThreads(MTL::Size(output.product.size(), 1, 1),
                                MTL::Size(assets.state->maxTotalThreadsPerThreadgroup(), 1, 1));
    cmdEncoder->endEncoding();

    return Result::SUCCESS;
}

template class MultiplyConstant<Device::Metal, CF32>;
template class MultiplyConstant<Device::Metal, F32>;

}  // namespace Jetstream
