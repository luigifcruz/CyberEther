#include "jstcore/fft/cpu.hpp"

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

static inline float scale(const float x, const float min, const float max) {
    return (x - min) / (max - min);
}

static inline float amplt(const std::complex<float> x, const int n) {
    return 20 * log10(abs(x) / n);
}

CPU::CPU(const Config& config, const Input& input) : Generic(config, input) {
    if ((input.in.location & Locale::CPU) != Locale::CPU) {
        std::cerr << "[FFT::CPU] This implementation expects a Locale::CPU input." << std::endl;
        JST_CHECK_THROW(Result::ERROR);
    }

    auto n = input.in.buf.size();
    fft_in.resize(n);
    fft_out.resize(n);
    amp_out.resize(n);

    cf_plan = fftwf_plan_dft_1d(input.in.buf.size(), reinterpret_cast<fftwf_complex*>(fft_in.data()),
            reinterpret_cast<fftwf_complex*>(fft_out.data()), FFTW_FORWARD, FFTW_MEASURE);

    out.buf = amp_out;
    out.location = Locale::CPU;
}

FFT::CPU::~CPU() {
    fftwf_destroy_plan(cf_plan);
}

const Result CPU::underlyingCompute() {
    float tmp;
    auto n = fft_in.size();
    auto [min, max] = config.amplitude;

    for (size_t i = 0; i < n; i++) {
        fft_in[i] = input.in.buf[i] * window[i];
    }

    fftwf_execute(cf_plan);

    for (size_t i = 0; i < n; i++) {
        tmp = amplt(fft_out[i], n);
        tmp = scale(tmp, min, max);
        tmp = std::clamp(tmp, 0.0f, 1.0f);

        amp_out[i] = tmp;
    }

    return Result::SUCCESS;
}

} // namespace Jetstream::FFT
