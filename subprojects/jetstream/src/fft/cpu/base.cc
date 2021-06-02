#include "jetstream/fft/cpu.hpp"
#include <algorithm>

namespace Jetstream::FFT {

CPU::CPU(Config& c) : Generic(c) {
    buf_fft.resize(in.buf.size());
    buf_out.resize(in.buf.size());
    out.buf = buf_out;

    std::cout << "[JST:FFT:CPU]: FFTW Version: " << fftwf_version << std::endl;

    cf_plan = fftwf_plan_dft_1d(in.buf.size(), reinterpret_cast<fftwf_complex*>(in.buf.data()),
            reinterpret_cast<fftwf_complex*>(buf_fft.data()), FFTW_FORWARD, FFTW_MEASURE);
}

CPU::~CPU() {
    fftwf_destroy_plan(cf_plan);
}

Result CPU::underlyingCompute() {
    fftwf_execute(cf_plan);

    for (int i = 0; i < buf_fft.size(); i++) {
        int ix;
        int stride = buf_fft.size() / 2;

        if (i < stride) {
            ix = stride + i;
        } else {
            ix = i - stride;
        }

        buf_out[i] = 20 * log10(abs(buf_fft[ix]) / buf_fft.size()) / cfg.min_db;

        if (buf_out[i] < -1.0) {
            buf_out[i] = -1.0;
        }

        if (buf_out[i] > 0.0) {
            buf_out[i] = 0.0;
        }
    }

    return Result::SUCCESS;
}

 Result CPU::underlyingPresent() {
    return Result::SUCCESS;
}

} // namespace Jetstream::FFT
