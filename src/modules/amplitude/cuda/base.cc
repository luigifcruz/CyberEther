#include "../generic.cc"

#include "jetstream/memory/devices/cuda/copy.hh"

namespace Jetstream {

template<Device D, typename IT, typename OT>
struct Amplitude<D, IT, OT>::Impl {
    std::vector<U64> grid;
    std::vector<U64> block;

    std::vector<void*> arguments;

    Tensor<Device::CUDA, IT> input;
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
    JST_TRACE("Create Amplitude compute core using CUDA backend.");

    // Create CUDA kernel.

    ctx.cuda->createKernel("amplitude", R"""(
        __global__ void amplitude(const float2* input, float* output, float scalingCoeff, size_t size) {
            size_t id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < size) {
                float2 number = input[id];
                float real = number.x;
                float imag = number.y;
                float pwr = sqrtf((real * real) + (imag * imag));
                output[id] = 20.0f * log10f(pwr) + scalingCoeff;
            }
        }
    )""");

    // Initialize kernel size.

    static U64 numElements = input.buffer.size();
    U64 threadsPerBlock = 256;
    U64 blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    pimpl->grid = { blocksPerGrid, 1, 1 };
    pimpl->block = { threadsPerBlock, 1, 1 };

    // Initialize kernel input.

    if (!input.buffer.device_native()) {
        pimpl->input = Tensor<Device::CUDA, IT>(input.buffer.shape());
    } else {
        pimpl->input = input.buffer;
    }

    // Initialize kernel arguments.

    pimpl->arguments = {
        pimpl->input.data_ptr(),
        output.buffer.data_ptr(),
        &scalingCoeff,
        &numElements,
    };

    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
Result Amplitude<D, IT, OT>::compute(const Context& ctx) {
    if (!input.buffer.device_native()) {
        JST_CHECK(Memory::Copy(pimpl->input, input.buffer, ctx.cuda->stream()));
    }

    JST_CHECK(ctx.cuda->launchKernel("amplitude", 
                                     pimpl->grid, 
                                     pimpl->block, 
                                     pimpl->arguments.data()));

    return Result::SUCCESS;
}

JST_AMPLITUDE_CUDA(JST_INSTANTIATION)
JST_AMPLITUDE_CUDA(JST_BENCHMARK)

}  // namespace Jetstream
