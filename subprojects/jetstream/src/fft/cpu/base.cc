#include "jetstream/fft/cpu.hpp"

namespace Jetstream::FFT {

CPU::CPU(Config& c, DF::CPU::CF32V& d) : Generic(c), df(d) {
    df.output = std::make_shared<std::vector<std::complex<float>>>();
    df.output->resize(df.input->size());

    std::cout << "[JST:FFT:CPU]: FFTW Version: " << fftwf_version << std::endl;

    cf_plan = fftwf_plan_dft_1d(df.input->size(), reinterpret_cast<fftwf_complex*>(df.input->data()),
            reinterpret_cast<fftwf_complex*>(df.output->data()), FFTW_FORWARD, FFTW_MEASURE);
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
