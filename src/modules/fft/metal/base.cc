#include "../generic.cc"

namespace Jetstream {

template<>
Result FFT<Device::Metal, CF32>::createCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Create FFT compute core using Metal backend.");

    auto& assets = metal;
    auto& runtime = meta.metal;

    // Assign buffers to module assets.
    assets.input = input.buffer;
    assets.output = output.buffer;

    // Create VkFFT instance.
    assets.app = new VkFFTApplication({});
    assets.configuration = new VkFFTConfiguration({});
    assets.configuration->FFTdim = 1;
    assets.configuration->size[0] = (config.offset != 0 || config.size != 0) ? config.size : input.buffer.shape()[1];
    assets.configuration->device = Backend::State<Device::Metal>()->getDevice();
    assets.configuration->queue = runtime.commandQueue;
    assets.configuration->doublePrecision = false;
    assets.configuration->numberBatches = input.buffer.shape()[0];
    assets.configuration->isInputFormatted = 1;
    assets.configuration->inputBufferSize = new U64(input.buffer.size_bytes());
    assets.configuration->inputBuffer = const_cast<MTL::Buffer**>(&assets.input);
    assets.configuration->bufferSize = new U64(output.buffer.size_bytes());
    assets.configuration->buffer = &assets.output;
    assets.configuration->inputBufferOffset = config.offset * sizeof(CF32);

    if (auto res = initializeVkFFT(assets.app, *assets.configuration); res != VKFFT_SUCCESS) {
        JST_ERROR("Failed to initialize VkFFT: {}", static_cast<int>(res));
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

template<>
Result FFT<Device::Metal, CF32>::destroyCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Destroy FFT compute core using Metal backend.");

    auto& assets = metal;

    if (!assets.app) {
        return Result::SUCCESS;
    }

    deleteVkFFT(assets.app);
  
    free(assets.configuration->inputBufferSize);
    free(assets.configuration->bufferSize);
    free(assets.configuration);
    free(assets.app);

    return Result::SUCCESS;
}

template<>
Result FFT<Device::Metal, CF32>::compute(const RuntimeMetadata& meta) {
    auto& assets = metal;
    auto& runtime = meta.metal;

    VkFFTLaunchParams launchParams = {};
    launchParams.commandBuffer = runtime.commandBuffer;
    launchParams.commandEncoder = runtime.commandBuffer->computeCommandEncoder();

    const int inverse = static_cast<int>(config.forward);
    if (auto res = VkFFTAppend(assets.app, inverse, &launchParams); res != VKFFT_SUCCESS) {
        JST_ERROR("Failed to append to VkFFT: {}", static_cast<int>(res));
        return Result::ERROR;
    }

    launchParams.commandEncoder->endEncoding();

    return Result::SUCCESS;
}

template class FFT<Device::Metal, CF32>;
    
}  // namespace Jetstream