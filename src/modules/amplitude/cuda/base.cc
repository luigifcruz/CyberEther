#include "../generic.cc"

#include "jetstream/memory2/helpers.hh"

namespace Jetstream {

template<Device D, typename IT, typename OT>
struct Amplitude<D, IT, OT>::Impl {
    std::vector<U64> grid;
    std::vector<U64> block;

    std::vector<void*> arguments;

    mem2::Tensor input;

    F32 scalingCoeff = 0.0f;
    U64 numberOfElements = 0;
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

    if constexpr (std::is_same_v<IT, CF32> && std::is_same_v<OT, F32>) {
        ctx.cuda->createKernel("amplitude", R"""(
            __global__ void amplitude(const float2* input, float* output, float scalingCoeff, size_t size) {
                size_t id = blockIdx.x * blockDim.x + threadIdx.x;
                if (id < size) {
                    float2 number = input[id];
                    float real = number.x;
                    float imag = number.y;
                    float pwr = fmaxf(sqrtf((real * real) + (imag * imag)), 1e-20f);
                    output[id] = 20.0f * log10f(pwr) + scalingCoeff;
                }
            }
        )""");
    } else if constexpr (std::is_same_v<IT, F32> && std::is_same_v<OT, F32>) {
        ctx.cuda->createKernel("amplitude", R"""(
            __global__ void amplitude(const float* input, float* output, float scalingCoeff, size_t size) {
                size_t id = blockIdx.x * blockDim.x + threadIdx.x;
                if (id < size) {
                    float pwr = fmaxf(fabs(input[id]), 1e-20f);
                    output[id] = 20.0f * log10f(pwr) + scalingCoeff;
                }
            }
        )""");
    }

    // Initialize kernel size.

    U64 threadsPerBlock = 512;
    U64 blocksPerGrid = (pimpl->numberOfElements + threadsPerBlock - 1) / threadsPerBlock;

    pimpl->grid = { blocksPerGrid, 1, 1 };
    pimpl->block = { threadsPerBlock, 1, 1 };

    // Initialize kernel input.

    if (!input.buffer.device_native() && input.buffer.contiguous()) {
        pimpl->input = mem2::Tensor(input.buffer.shape(), mem2::DataType::fromCppType<IT>(), Device::CUDA);
    } else {
        pimpl->input = input.buffer;
    }

    // Initialize kernel arguments.

    pimpl->arguments = {
        pimpl->input.data_ptr(),
        output.buffer.data_ptr(),
        &pimpl->scalingCoeff,
        &pimpl->numberOfElements,
    };

    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
Result Amplitude<D, IT, OT>::compute(const Context& ctx) {
    if (!input.buffer.device_native() && input.buffer.contiguous()) {
        JST_CHECK(pimpl->input.copy_from(input.buffer));
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
