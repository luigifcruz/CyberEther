#include "jetstream/modules/fft.hh"

namespace Jetstream {

template<>
FFT<Device::CPU>::FFT(const Config& config, const Input& input) 
    : config(config), input(input) {
    JST_DEBUG("Initializing FFT module with CPU backend.");

    // Intialize output.
    this->InitInput(this->input.buffer, getBufferSize());
    this->InitOutput(this->output.buffer, getBufferSize());

    // Check parameters. 
    if (this->input.buffer.size() != this->config.size) {
        JST_FATAL("Input Buffer size ({}) is different than the" \
            "configuration size ({}).", this->input.buffer.size(),
            this->config.size);
        throw Result::ERROR;
    }

    // Generate FFT plan.
    auto inBuf = reinterpret_cast<fftwf_complex*>(input.buffer.data());
    auto outBuf = reinterpret_cast<fftwf_complex*>(output.buffer.data());
    auto direction = (config.direction == Direction::Forward) ? FFTW_FORWARD : FFTW_BACKWARD;
    CPU.fftPlan = fftwf_plan_dft_1d(config.size, inBuf, outBuf, direction, FFTW_MEASURE);

    JST_INFO("===== FFT Module Configuration");
    JST_INFO("Size: {}", this->config.size);
    JST_INFO("Direction: {}", static_cast<I64>(config.direction));
}

template<>
const Result FFT<Device::CPU>::compute() {
    fftwf_execute(CPU.fftPlan);
    return Result::SUCCESS;
}
    
}  // namespace Jetstream
