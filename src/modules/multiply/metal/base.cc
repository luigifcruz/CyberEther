#include "../generic.cc"

#include "jetstream/memory2/helpers.hh"

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
struct Multiply<D, T>::Impl {
    mem2::Tensor a;
    mem2::Tensor b;
    mem2::Tensor c;

    MTL::ComputePipelineState* state;
};

template<Device D, typename T>
Multiply<D, T>::Multiply() {
    impl = std::make_unique<Impl>();
}

template<Device D, typename T>
Multiply<D, T>::~Multiply() {
    impl.reset();
}

template<Device D, typename T>
Result Multiply<D, T>::createCompute(const Context&) {
    JST_TRACE("Create Multiply compute core using Metal backend.");

    JST_CHECK(Metal::CompileKernel(shadersSrc, "multiply", &impl->state));

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Multiply<D, T>::compute(const Context& ctx) {
    // TODO: Implement new multiplication logic.

    auto cmdEncoder = ctx.metal->commandBuffer()->computeCommandEncoder();
    cmdEncoder->setComputePipelineState(impl->state);
    cmdEncoder->setBuffer(input.factorA.data(), 0, 0);
    cmdEncoder->setBuffer(input.factorB.data(), 0, 1);
    cmdEncoder->setBuffer(output.product.data(), 0, 2);
    cmdEncoder->dispatchThreads(MTL::Size(output.product.size(), 1, 1),
                                MTL::Size(impl->state->maxTotalThreadsPerThreadgroup(), 1, 1));
    cmdEncoder->endEncoding();

    return Result::SUCCESS;
}

JST_MULTIPLY_METAL(JST_INSTANTIATION)
JST_MULTIPLY_METAL(JST_BENCHMARK)

}  // namespace Jetstream
