#include "../generic.cc"

#include "jetstream/memory/devices/cuda/copy.hh"

namespace Jetstream {

template<Device D, typename T>
struct Lineplot<D, T>::Impl {
    std::vector<U64> grid;
    std::vector<U64> block;

    std::vector<void*> argumentsLineplot;
    std::vector<void*> argumentsThickness;

    Tensor<Device::CUDA, T> input;
    Tensor<Device::CUDA, T> intermediate;
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

    ctx.cuda->createKernel("thickness", R"""(
        __device__ inline float2 ComputePerpendicular(float2 d, float2 thickness) {
            // Compute length
            const float length = sqrtf(d.x * d.x + d.y * d.y);

            // Normalize
            d.x /= length;
            d.y /= length;

            // Return perperdicular (normalized)
            return make_float2(-d.y * thickness.x, d.x * thickness.y);
        }

        __global__ void thickness(const float2* input, float* output, size_t numberOfElements, float2 thickness) {
            size_t id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < numberOfElements - 1) {
                const float2 p1 = input[id + 0];
                const float2 p2 = input[id + 1];

                const float2 d = make_float2(p2.x - p1.x, p2.y - p1.y);
                const float2 perp = ComputePerpendicular(d, thickness);

                const size_t idx = id * 4 * 3;

                // Upper left
                output[idx] = p1.x + perp.x;
                output[idx + 1] = p1.y + perp.y;
                output[idx + 2] = 0.0f;

                // Lower left
                output[idx + 3] = p1.x - perp.x;
                output[idx + 4] = p1.y - perp.y;
                output[idx + 5] = 0.0f;
                
                // Upper right
                output[idx + 6] = p2.x + perp.x;
                output[idx + 7] = p2.y + perp.y;
                output[idx + 8] = 0.0f;

                // Lower right
                output[idx + 9] = p2.x - perp.x;
                output[idx + 10] = p2.y - perp.y;
                output[idx + 11] = 0.0f;
            }
        }
    )""");

    // Initialize kernel size.

    U64 threadsPerBlock = 256;
    U64 blocksPerGrid = (numberOfElements + threadsPerBlock - 1) / threadsPerBlock;

    pimpl->grid = { blocksPerGrid, 1, 1 };
    pimpl->block = { threadsPerBlock, 1, 1 };

    // Initialize kernel input.

    if (!input.buffer.device_native()) {
        pimpl->input = Tensor<Device::CUDA, T>(input.buffer.shape());
    } else {
        pimpl->input = input.buffer;
    }

    // Allocate intermediate memory.

    pimpl->intermediate = Tensor<Device::CUDA, T>({numberOfElements, 2});
    JST_CUDA_CHECK(cudaMemset(pimpl->intermediate.data(), 0, pimpl->intermediate.size_bytes()), [&]{
        JST_ERROR("Failed to initialize intermediate memory: {}", err);
    });

    // Initialize kernel arguments.

    pimpl->argumentsLineplot = {
        pimpl->input.data_ptr(),
        pimpl->intermediate.data_ptr(),
        &normalizationFactor,
        &numberOfBatches,
        &numberOfElements,
        &config.averaging,
    };

    pimpl->argumentsThickness = {
        pimpl->intermediate.data_ptr(),
        signal.data_ptr(),
        &numberOfElements,
        &thickness,
    };

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Lineplot<D, T>::compute(const Context& ctx) {
    if (!input.buffer.device_native()) {
        JST_CHECK(Memory::Copy(pimpl->input, input.buffer, ctx.cuda->stream()));
    }

    // TODO: Join kernels.

    JST_CHECK(ctx.cuda->launchKernel("lineplot", 
                                     pimpl->grid, 
                                     pimpl->block, 
                                     pimpl->argumentsLineplot.data()));
    
    JST_CHECK(ctx.cuda->launchKernel("thickness",
                                     pimpl->grid,
                                     pimpl->block,
                                     pimpl->argumentsThickness.data()));

    return Result::SUCCESS;
}

JST_LINEPLOT_CUDA(JST_INSTANTIATION)
JST_LINEPLOT_CUDA(JST_BENCHMARK)

}  // namespace Jetstream
