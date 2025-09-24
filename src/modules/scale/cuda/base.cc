#include "../generic.cc"

#include "jetstream/memory2/helpers.hh"

namespace Jetstream {

template<Device D, typename T>
struct Scale<D, T>::Impl {
    std::vector<U64> grid;
    std::vector<U64> block;

    std::vector<void*> arguments;

    mem2::Tensor input;

    F32 scalingCoeff;
    F32 offsetCoeff;
    U64 numberOfElements;
};

template<Device D, typename T>
Scale<D, T>::Scale() {
    impl = std::make_unique<Impl>();
}

template<Device D, typename T>
Scale<D, T>::~Scale() {
    impl.reset();
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
    U64 blocksPerGrid = (impl->numberOfElements + threadsPerBlock - 1) / threadsPerBlock;

    impl->grid = { blocksPerGrid, 1, 1 };
    impl->block = { threadsPerBlock, 1, 1 };

    // Initialize kernel input.

    if (!input.buffer.device_native() && input.buffer.contiguous()) {
        impl->input = mem2::Tensor(input.buffer.shape(), mem2::DataType::fromCppType<T>(), Device::CUDA);
    } else {
        impl->input = input.buffer;
    }

    // Initialize kernel arguments.

    impl->arguments = {
        impl->input.data_ptr(),
        output.buffer.data_ptr(),
        &impl->scalingCoeff,
        &impl->offsetCoeff,
        &impl->numberOfElements,
    };

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Scale<D, T>::compute(const Context& ctx) {
    if (!input.buffer.device_native() && input.buffer.contiguous()) {
        JST_CHECK(impl->input.copy_from(input.buffer));
    }

    JST_CHECK(ctx.cuda->launchKernel("scale",
                                     impl->grid,
                                     impl->block,
                                     impl->arguments.data()));

    return Result::SUCCESS;
}

JST_SCALE_CUDA(JST_INSTANTIATION)
JST_SCALE_CUDA(JST_BENCHMARK)

}  // namespace Jetstream
