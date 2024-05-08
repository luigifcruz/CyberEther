#include "../generic.cc"

#include "jetstream/memory/devices/cuda/copy.hh"

namespace Jetstream {

template<Device D, typename T>
struct Lineplot<D, T>::Impl {
    std::vector<U64> grid;
    std::vector<U64> block;

    std::vector<void*> argumentsLineplot;

    Tensor<Device::CUDA, T> input;
};

template<Device D, typename T>
Lineplot<D, T>::Lineplot() {
    pimpl = std::make_unique<Impl>();
    gimpl = std::make_unique<GImpl>();
}

template<Device D, typename T>
Lineplot<D, T>::~Lineplot() {
    pimpl.reset();
    gimpl.reset();
}

template<Device D, typename T>
Result Lineplot<D, T>::createCompute(const Context& ctx) {
    JST_TRACE("Create Lineplot compute core using CUDA backend.");

    // Create CUDA kernel.

    ctx.cuda->createKernel("lineplot", R"""(
        __global__ void lineplot(const float* input, float2* output, float normalizationFactor, size_t numberOfBatches, size_t numberOfElements, size_t averaging) {
            size_t id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < numberOfElements) {
                // Compute average amplitude within a batch.
                float amplitude = 0.0f;
                for (size_t i = 0; i < numberOfBatches; ++i) {
                    amplitude += input[id + (i * numberOfElements)];
                }
                amplitude = (amplitude * normalizationFactor) - 1.0f;

                // Calculate moving average.
                float average = output[id].y;
                average -= average / averaging;
                average += amplitude / averaging;

                // Store result.
                output[id].x = id * 2.0f / (numberOfElements - 1) - 1.0f;
                output[id].y = average;
            }
        }
    )""");

    // Initialize kernel size.

    U64 threadsPerBlock = 256;
    U64 blocksPerGrid = (numberOfElements + threadsPerBlock - 1) / threadsPerBlock;

    pimpl->grid = { blocksPerGrid, 1, 1 };
    pimpl->block = { threadsPerBlock, 1, 1 };

    // Initialize kernel input.

    if (!input.buffer.device_native() && input.buffer.contiguous()) {
        pimpl->input = Tensor<Device::CUDA, T>(input.buffer.shape());
    } else {
        pimpl->input = input.buffer;
    }

    // Initialize kernel arguments.

    pimpl->argumentsLineplot = {
        pimpl->input.data_ptr(),
        signalPoints.data_ptr(),
        &normalizationFactor,
        &numberOfBatches,
        &numberOfElements,
        &config.averaging,
    };

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Lineplot<D, T>::compute(const Context& ctx) {
    if (!input.buffer.device_native() && input.buffer.contiguous()) {
        JST_CHECK(Memory::Copy(pimpl->input, input.buffer, ctx.cuda->stream()));
    }

    // TODO: Join kernels.

    JST_CHECK(ctx.cuda->launchKernel("lineplot", 
                                     pimpl->grid, 
                                     pimpl->block, 
                                     pimpl->argumentsLineplot.data()));

    updateSignalPointsFlag = true;

    return Result::SUCCESS;
}

JST_LINEPLOT_CUDA(JST_INSTANTIATION)
JST_LINEPLOT_CUDA(JST_BENCHMARK)

}  // namespace Jetstream
