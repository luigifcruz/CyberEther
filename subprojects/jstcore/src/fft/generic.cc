#include "jstcore/fft/generic.hpp"

namespace Jetstream::FFT {

Generic::Generic(const Config & config, const Input & input) : config(config), input(input) {
    if (input.in.location != Locale::CPU) {
        throw Result::ERROR;
    }

    auto n = input.in.buf.size();
    window.resize(input.in.buf.size());

    float tap;
    for (size_t i = 0; i < n; i++) {
        tap = 0.5 * (1 - cos(2 * M_PI * i / n));
        tap = (i % 2) == 0 ? tap : -tap;
        window[i] = std::complex<float>(tap, 0.0);
    }
}

Result Generic::compute() {
    return this->underlyingCompute();
}

Range<float> Generic::amplitude(const Range<float> & ampl) {
    config.amplitude = ampl;
    return this->amplitude();
}

} // namespace Jetstream::FFT
