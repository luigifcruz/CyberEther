#include "jetstream/modules/fft.hh"

namespace Jetstream {

template<>
const Result FFT<Device::CPU, CF32>::generatePlanCPU() {
    auto inBuf = reinterpret_cast<fftwf_complex*>(input.buffer.data());
    auto outBuf = reinterpret_cast<fftwf_complex*>(output.buffer.data());
    auto direction = (config.direction == Direction::Forward) ? FFTW_FORWARD : FFTW_BACKWARD;
    CPU.fftPlanCF32 = fftwf_plan_dft_1d(config.size, inBuf, outBuf, direction, FFTW_MEASURE);
    return Result::SUCCESS;
}

template<>
const Result FFT<Device::CPU, CF64>::generatePlanCPU() {
    auto inBuf = reinterpret_cast<fftw_complex*>(input.buffer.data());
    auto outBuf = reinterpret_cast<fftw_complex*>(output.buffer.data());
    auto direction = (config.direction == Direction::Forward) ? FFTW_FORWARD : FFTW_BACKWARD;
    CPU.fftPlanCF64 = fftw_plan_dft_1d(config.size, inBuf, outBuf, direction, FFTW_MEASURE);
    return Result::SUCCESS;
}

template<Device D, typename T>
FFT<D, T>::FFT(const Config& config, const Input& input) 
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
    JST_CHECK_THROW(this->generatePlanCPU());

    JST_INFO("===== FFT Module Configuration");
    JST_INFO("Size: {}", this->config.size);
    JST_INFO("Direction: {}", static_cast<I64>(config.direction));
    JST_INFO("Input Type: {}", getTypeName<T>());
}

template<>
const Result FFT<Device::CPU, CF32>::compute(const RuntimeMetadata& meta) {
    fftwf_execute(CPU.fftPlanCF32);
    return Result::SUCCESS;
}

template<>
const Result FFT<Device::CPU, CF64>::compute(const RuntimeMetadata& meta) {
    fftw_execute(CPU.fftPlanCF64);
    return Result::SUCCESS;
}

template class FFT<Device::CPU, CF32>;
template class FFT<Device::CPU, CF64>;
    
}  // namespace Jetstream
