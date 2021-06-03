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

        buf_out[i] = 20 * log10f_fast(abs(buf_fft[ix]) / buf_fft.size());
        buf_out[i] = (buf_out[i] - cfg.min_db)/(cfg.max_db - cfg.min_db);

        if (buf_out[i] < 0.0) {
            buf_out[i] = 0.0;
        }

        if (buf_out[i] > 1.0) {
            buf_out[i] = 1.0;
        }
    }

    return Result::SUCCESS;
}

 Result CPU::underlyingPresent() {
    return Result::SUCCESS;
}

// Faster Log10 by http://openaudio.blogspot.com/2017/02/faster-log10-and-pow.html
inline float CPU::log10f_fast(float X) {
  float Y, F;
  int E;
  F = frexpf(fabsf(X), &E);
  Y = 1.23149591368684f;
  Y *= F;
  Y += -4.11852516267426f;
  Y *= F;
  Y += 6.02197014179219f;
  Y *= F;
  Y += -3.13396450166353f;
  Y += E;
  return Y * 0.3010299956639812f;
}

} // namespace Jetstream::FFT
