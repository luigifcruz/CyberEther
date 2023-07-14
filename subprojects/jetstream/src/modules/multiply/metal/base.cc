#include "../generic.cc"

namespace Jetstream {

static const char shadersSrc[] = R"""(
    #include <metal_stdlib>

    using namespace metal;

    kernel void multiply(constant const float *factorA [[ buffer(0) ]],
                         constant const float *factorB [[ buffer(1) ]],
                         device float *product [[ buffer(2) ]],
                         uint id[[ thread_position_in_grid ]]) {
        const uint index = id * 2;
        product[index + 0] = (factorA[index + 0] * (factorB[index + 0])) - 
                             (factorA[index + 1] * (factorB[index + 1]));
        product[index + 1] = (factorA[index + 0] * (factorB[index + 1])) + 
                             (factorA[index + 1] * (factorB[index + 0]));
    }
)""";

template<Device D, typename T>
Result Multiply<D, T>::createCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Create Multiply compute core using Metal backend.");

    auto& assets = metal;

    JST_CHECK(Metal::CompileKernel(shadersSrc, "multiply", &assets.state));

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Multiply<D, T>::compute(const RuntimeMetadata& meta) {
    auto& assets = metal;
    auto& runtime = meta.metal;
    
    auto cmdEncoder = runtime.commandBuffer->computeCommandEncoder();
    cmdEncoder->setComputePipelineState(assets.state);
    cmdEncoder->setBuffer(input.factorA, 0, 0);
    cmdEncoder->setBuffer(input.factorB, 0, 1);
    cmdEncoder->setBuffer(output.product, 0, 2);
    cmdEncoder->dispatchThreads(MTL::Size(output.product.size(), 1, 1),
                                MTL::Size(assets.state->maxTotalThreadsPerThreadgroup(), 1, 1));
    cmdEncoder->endEncoding();

    return Result::SUCCESS;
}


template class Multiply<Device::Metal, CF32>;
    
}  // namespace Jetstream
