#include "../generic.cc"

namespace Jetstream {

template<>
const Result FFT<Device::CPU, CF32>::createCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Create FFT compute core using CPU backend.");

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
