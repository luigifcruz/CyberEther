#include "jetstream/fft/cpu.hpp"

namespace Jetstream::FFT {

CPU::CPU(Config& c) : Generic(c) {
    cfg.output = std::make_shared<std::vector<std::complex<float>>>();
    cfg.output->resize(cfg.input->size());

    std::cout << "[JST:FFT:CPU]: FFTW Version: " << fftwf_version << std::endl;

    cf_plan = fftwf_plan_dft_1d(cfg.input->size(), reinterpret_cast<fftwf_complex*>(cfg.input->data()),
            reinterpret_cast<fftwf_complex*>(cfg.output->data()), FFTW_FORWARD, FFTW_MEASURE);
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
