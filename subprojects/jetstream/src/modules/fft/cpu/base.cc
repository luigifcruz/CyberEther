#include "jetstream/modules/fft.hh"

namespace Jetstream {

template<Device D, typename T>
FFT<D, T>::FFT(const Config& config,
               const Input& input) 
         : config(config), input(input) {
    JST_DEBUG("Initializing FFT module with CPU backend.");

    JST_CHECK_THROW(this->InitInput(this->input.buffer, getBufferSize()));
    JST_CHECK_THROW(this->InitOutput(this->output.buffer, getBufferSize()));

    // Check parameters. 
    if (this->input.buffer.size() != this->config.size) {
        JST_FATAL("Input Buffer size ({}) is different than the" \
            "configuration size ({}).", this->input.buffer.size(),
            this->config.size);
        JST_CHECK_THROW(Result::ERROR);
    }
}

template<Device D, typename T>
void FFT<D, T>::summary() const {
    JST_INFO("===== FFT Module Configuration");
    JST_INFO("Size: {}", this->config.size);
    JST_INFO("Direction: {}", static_cast<I64>(config.direction));
    JST_INFO("Input Type: {}", NumericTypeInfo<T>().name);
}

template<>
const Result FFT<Device::CPU, CF32>::createCompute(const RuntimeMetadata& meta) {
    auto inBuf = reinterpret_cast<fftwf_complex*>(input.buffer.data());
    auto outBuf = reinterpret_cast<fftwf_complex*>(output.buffer.data());
    auto direction = (config.direction == Direction::Forward) ? FFTW_FORWARD : FFTW_BACKWARD;
    cpu.fftPlanCF32 = fftwf_plan_dft_1d(config.size, inBuf, outBuf, direction, FFTW_MEASURE);

    return Result::SUCCESS;
}

template<>
const Result FFT<Device::CPU, CF32>::compute(const RuntimeMetadata& meta) {
    fftwf_execute(cpu.fftPlanCF32);

    return Result::SUCCESS;
}

template class FFT<Device::CPU, CF32>;
    
}  // namespace Jetstream
