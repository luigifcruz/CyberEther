#include "../generic.cc"

#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-compare"
#define VKFFT_BACKEND 5
#include "vkFFT.h"
#pragma GCC diagnostic pop

namespace Jetstream {

template<Device D, typename IT, typename OT>
struct FFT<D, IT, OT>::Impl {
    VkFFTApplication* app;
    VkFFTConfiguration* configuration;
    const MTL::Buffer* input;
    MTL::Buffer* output;
};

template<Device D, typename IT, typename OT>
FFT<D, IT, OT>::FFT() {
    pimpl = std::make_unique<Impl>();
}

template<Device D, typename IT, typename OT>
FFT<D, IT, OT>::~FFT() {
    pimpl.reset();
}

template<Device D, typename IT, typename OT>
Result FFT<D, IT, OT>::createCompute(const Context& ctx) {
    JST_TRACE("Create FFT compute core using Metal backend.");

    // Assign buffers to module assets.
    pimpl->input = input.buffer.data();
    pimpl->output = output.buffer.data();

    // Create VkFFT instance.
    pimpl->app = new VkFFTApplication({});
    pimpl->configuration = new VkFFTConfiguration({});
    pimpl->configuration->FFTdim = 1;
    pimpl->configuration->size[0] = numberOfElements;
    pimpl->configuration->device = Backend::State<Device::Metal>()->getDevice();
    pimpl->configuration->queue = ctx.metal->commandQueue();
    pimpl->configuration->doublePrecision = false;
    pimpl->configuration->numberBatches = numberOfOperations;
    pimpl->configuration->isInputFormatted = 1;
    pimpl->configuration->inputBufferSize = new U64(input.buffer.size_bytes());
    pimpl->configuration->inputBuffer = const_cast<MTL::Buffer**>(&pimpl->input);
    pimpl->configuration->bufferSize = new U64(output.buffer.size_bytes());
    pimpl->configuration->buffer = &pimpl->output;

    if (auto res = initializeVkFFT(pimpl->app, *pimpl->configuration); res != VKFFT_SUCCESS) {
        JST_ERROR("Failed to initialize VkFFT: {}", static_cast<int>(res));
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
Result FFT<D, IT, OT>::destroyCompute(const Context&) {
    JST_TRACE("Destroy FFT compute core using Metal backend.");

    if (!pimpl->app) {
        return Result::SUCCESS;
    }

    deleteVkFFT(pimpl->app);
  
    free(pimpl->configuration->inputBufferSize);
    free(pimpl->configuration->bufferSize);
    free(pimpl->configuration);
    free(pimpl->app);

    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
Result FFT<D, IT, OT>::compute(const Context& ctx) {
    VkFFTLaunchParams launchParams = {};
    launchParams.commandBuffer = ctx.metal->commandBuffer();
    launchParams.commandEncoder = ctx.metal->commandBuffer()->computeCommandEncoder();

    const int inverse = static_cast<int>(!config.forward);
    if (auto res = VkFFTAppend(pimpl->app, inverse, &launchParams); res != VKFFT_SUCCESS) {
        JST_ERROR("Failed to append to VkFFT: {}", static_cast<int>(res));
        return Result::ERROR;
    }

    launchParams.commandEncoder->endEncoding();

    return Result::SUCCESS;
}

JST_FFT_METAL(JST_INSTANTIATION)
JST_FFT_METAL(JST_BENCHMARK)

}  // namespace Jetstream
