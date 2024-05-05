#include "../generic.cc"

#include "jetstream/memory/devices/cuda/copy.hh"

namespace Jetstream {

template<Device D, typename T>
struct Spectrogram<D, T>::Impl {
    std::vector<U64> decayGrid;
    std::vector<U64> riseGrid;

    std::vector<U64> decayBlock;
    std::vector<U64> riseBlock;

    std::vector<void*> decayArguments;
    std::vector<void*> riseArguments;

    Tensor<Device::CUDA, T> input;
};

template<Device D, typename T>
Spectrogram<D, T>::Spectrogram() {
    pimpl = std::make_unique<Impl>();
    gimpl = std::make_unique<GImpl>();
}

template<Device D, typename T>
Spectrogram<D, T>::~Spectrogram() {
    pimpl.reset();
    gimpl.reset();
}

template<Device D, typename T>
Result Spectrogram<D, T>::createCompute(const Context& ctx) {
    JST_TRACE("Create Spectrogram compute core using CUDA backend.");

    // Initialize kernel input.

    if (!input.buffer.device_native() && input.buffer.contiguous()) {
        pimpl->input = Tensor<Device::CUDA, T>(input.buffer.shape());
    } else {
        pimpl->input = input.buffer;
    }

    // Create CUDA kernel.

    ctx.cuda->createKernel("decay", R"""(
        __global__ void decay(float* bins, float decayFactor, size_t size) {
            const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < size) {
                bins[id] *= decayFactor;
            }
        }
    )""");

    ctx.cuda->createKernel("rise", R"""(
        __global__ void rise(const float* input, float* bins, size_t numberOfElements, size_t numberOfBatches, size_t height) {
            const size_t id = blockIdx.x * blockDim.x + threadIdx.x;

            if (id >= numberOfElements) {
                return;
            }

            for (size_t b = 0; b < numberOfBatches * numberOfElements; b += numberOfElements) {
                const size_t index = input[id + b] * height;

                if (index < height && index > 0) {
                    atomicAdd(&bins[id + (index * numberOfElements)], 0.02f);
                }
            }
        }
    )""");

    // Initialize kernel size.

    {
        U64 threadsPerBlock = 512;
        U64 blocksPerGrid = (totalFrequencyBins + threadsPerBlock - 1) / threadsPerBlock;

        pimpl->decayGrid = { blocksPerGrid, 1, 1 };
        pimpl->decayBlock = { threadsPerBlock, 1, 1 };
    }

    {
        U64 threadsPerBlock = 512;
        U64 blocksPerGrid = (numberOfElements + threadsPerBlock - 1) / threadsPerBlock;

        pimpl->riseGrid = { blocksPerGrid, 1, 1 };
        pimpl->riseBlock = { threadsPerBlock, 1, 1 };
    }

    // Initialize kernel arguments.

    pimpl->decayArguments = {
        frequencyBins.data_ptr(),
        &decayFactor,
        &totalFrequencyBins,
    };

    pimpl->riseArguments = {
        input.buffer.data_ptr(),
        frequencyBins.data_ptr(),
        &numberOfElements,
        &numberOfBatches,
        &config.height,
    };

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Spectrogram<D, T>::compute(const Context& ctx) {
    if (!input.buffer.device_native() && input.buffer.contiguous()) {
        JST_CHECK(Memory::Copy(pimpl->input, input.buffer, ctx.cuda->stream()));
    }

    JST_CHECK(ctx.cuda->launchKernel("decay", 
                                     pimpl->decayGrid, 
                                     pimpl->decayBlock, 
                                     pimpl->decayArguments.data()));

    JST_CHECK(ctx.cuda->launchKernel("rise", 
                                     pimpl->riseGrid, 
                                     pimpl->riseBlock, 
                                     pimpl->riseArguments.data()));

    return Result::SUCCESS;
}

JST_SPECTROGRAM_CUDA(JST_INSTANTIATION)
JST_SPECTROGRAM_CUDA(JST_BENCHMARK)

}  // namespace Jetstream
