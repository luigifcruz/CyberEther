#include "jetstream/fft/cpu.hpp"

namespace Jetstream::FFT {

CPU::CPU(Config& c, I& i) : Generic(c), input(i) {
    output.data.resize(input.data.size());

    std::cout << "[JST:FFT:CPU]: FFTW Version: " << fftwf_version << std::endl;

    cf_plan = fftwf_plan_dft_1d(input.data.size(), reinterpret_cast<fftwf_complex*>(input.data.data()),
            reinterpret_cast<fftwf_complex*>(output.data.data()), FFTW_FORWARD, FFTW_MEASURE);
}

CPU::~CPU() {
    fftwf_destroy_plan(cf_plan);
}

Result CPU::underlyingCompute() {
    fftwf_execute(cf_plan);
    return Result::SUCCESS;
}

Result CPU::underlyingPresent() {
    return Result::SUCCESS;
}

} // namespace Jetstream::FFT
