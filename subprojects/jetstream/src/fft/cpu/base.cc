#include "jetstream/fft/cpu.hpp"
#include <algorithm>

namespace Jetstream::FFT {

// Faster Log10 by http://openaudio.blogspot.com/2017/02/faster-log10-and-pow.html
static inline float log10(float X) {
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

static inline float clamp(float x, float a, float b) {
    return fmax(a, fmin(b, x));
}

static inline float scale(float x, float min, float max) {
    return (x - min) / (max - min);
}

static inline float amplt(std::complex<float> x, int n) {
    return 20 * log10(abs(x) / n);
}

static inline int shift(int i, unsigned int n) {
    return (i + (n / 2) - 1) % n;
}

CPU::CPU(Config& c) : Generic(c) {
    auto n = in.buf.size();
    fft_in.resize(n);
    fft_out.resize(n);
    amp_out.resize(n);
    out.buf = amp_out;

    cf_plan = fftwf_plan_dft_1d(in.buf.size(), reinterpret_cast<fftwf_complex*>(fft_in.data()),
            reinterpret_cast<fftwf_complex*>(fft_out.data()), FFTW_FORWARD, FFTW_ESTIMATE);
}

CPU::~CPU() {
    fftwf_destroy_plan(cf_plan);
}

Result CPU::underlyingCompute() {
    auto n = fft_in.size();
    for (int i = 0; i < n; i++) {
        fft_in[i] = in.buf[i];
    }

    fftwf_execute(cf_plan);

    for (int i = 0; i < n; i++) {
        float tmp;

        tmp = amplt(fft_out[shift(i, n)], n);
        tmp = scale(tmp, cfg.min_db, cfg.max_db);
        tmp = clamp(tmp, 0.0, 1.0);

        amp_out[i] = tmp;
    }

    return Result::SUCCESS;
}

Result CPU::underlyingPresent() {
    return Result::SUCCESS;
}


} // namespace Jetstream::FFT
