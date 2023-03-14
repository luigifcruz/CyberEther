#include "../generic.cc"

namespace Jetstream {

template<>
const Result FFT<Device::Metal, CF32>::createCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Create FFT compute core using Metal backend.");

    // auto inBuf = reinterpret_cast<fftwf_complex*>(input.buffer.data());
    // auto outBuf = reinterpret_cast<fftwf_complex*>(output.buffer.data());
    // auto direction = (config.direction == Direction::Forward) ? FFTW_FORWARD : FFTW_BACKWARD;
    // cpu.fftPlanCF32 = fftwf_plan_dft_1d(config.size, inBuf, outBuf, direction, FFTW_MEASURE);


    // TODO: Leak.
    metal.app = new VkFFTApplication({});

    VkFFTConfiguration configuration = {};
    configuration.FFTdim = 1;
    configuration.size[0] = input.buffer.size();
    configuration.device = Backend::State<Device::Metal>()->getDevice();
    configuration.queue = meta.metal.commandQueue;
    configuration.doublePrecision = false;
    configuration.numberBatches = 1;

    uint64_t bufferSize = (uint64_t)sizeof(float) * 2 * configuration.size[0] * configuration.numberBatches;
    MTL::Buffer* buffer = 0;
    buffer = configuration.device->newBuffer(bufferSize, MTL::ResourceStorageModePrivate);
    configuration.buffer = &buffer;

    // TODO: Leak.
    configuration.bufferSize = new U64(bufferSize);

    configuration.isInputFormatted = 1;
    configuration.isOutputFormatted = 1;
    // TODO: Leak.
    configuration.inputBufferSize = new U64(bufferSize);
    configuration.outputBufferSize = new U64(bufferSize);
    
    metal.input = input.buffer.buffer();
    metal.output = output.buffer.buffer();
    configuration.inputBuffer = &metal.input;
    configuration.outputBuffer = &metal.output;

    auto res = initializeVkFFT(metal.app, configuration);
    if (res != VKFFT_SUCCESS) {
        JST_FATAL("{}", res);
        throw res;
    }

    return Result::SUCCESS;
}

template<>
const Result FFT<Device::Metal, CF32>::compute(const RuntimeMetadata& meta) {
    VkFFTLaunchParams launchParams = {};
    launchParams.commandBuffer = meta.metal.commandBuffer;
    auto cmdEncoder = meta.metal.commandBuffer->computeCommandEncoder();
    launchParams.commandEncoder = cmdEncoder;
    auto res = VkFFTAppend(metal.app, 0, &launchParams);
    if (res != VKFFT_SUCCESS) {
        JST_FATAL("{}", res);
        throw res;
    }
    cmdEncoder->endEncoding();

    return Result::SUCCESS;
}

template class FFT<Device::Metal, CF32>;
    
}  // namespace Jetstream
