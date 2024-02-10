#include "../generic.cc"

#include "jetstream/memory/devices/cuda/copy.hh"

namespace Jetstream {

template<Device D, typename T>
struct Scale<D, T>::Impl {
    std::vector<U64> grid;
    std::vector<U64> block;

    std::vector<void*> arguments;

    Tensor<Device::CUDA, T> input;
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
    JST_TRACE("Create Scale compute core using CUDA backend.");

    // Create CUDA kernel.

    ctx.cuda->createKernel("scale", R"""(
        __global__ void scale(const float* input, float* output, float scalingCoeff, float offsetCoeff, size_t size) {
            size_t id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < size) {
                output[id] = input[id] * scalingCoeff + offsetCoeff;
            }
        }
    )""");

    // Initialize kernel size.

    U64 threadsPerBlock = 512;
    U64 blocksPerGrid = (numberOfElements + threadsPerBlock - 1) / threadsPerBlock;

    pimpl->grid = { blocksPerGrid, 1, 1 };
    pimpl->block = { threadsPerBlock, 1, 1 };

    // Initialize kernel input.

    if (!input.buffer.device_native()) {
        pimpl->input = Tensor<Device::CUDA, T>(input.buffer.shape());
    } else {
        pimpl->input = input.buffer;
    }

    // Initialize kernel arguments.

    pimpl->arguments = {
        pimpl->input.data_ptr(),
        output.buffer.data_ptr(),
        &scalingCoeff,
        &offsetCoeff,
        &numberOfElements,
    };

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Scale<D, T>::compute(const Context& ctx) {
    if (!input.buffer.device_native()) {
        JST_CHECK(Memory::Copy(pimpl->input, input.buffer, ctx.cuda->stream()));
    }

    JST_CHECK(ctx.cuda->launchKernel("scale", 
                                     pimpl->grid, 
                                     pimpl->block, 
                                     pimpl->arguments.data()));

    return Result::SUCCESS;
}

JST_SCALE_CUDA(JST_INSTANTIATION)
JST_SCALE_CUDA(JST_BENCHMARK)

}  // namespace Jetstream
