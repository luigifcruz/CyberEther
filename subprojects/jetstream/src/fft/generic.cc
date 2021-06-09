#include "jetstream/fft/generic.hpp"

namespace Jetstream::FFT {

Generic::Generic(Config& c) : Module(c.policy), cfg(c), in(c.input0) {
    auto n = in.buf.size();
    window.resize(in.buf.size());
    for (int i = 0; i < n; i++) {
        float tap = 0.5 * (1 - cos(2 * M_PI * i / n));
        window[i] = std::complex<float>(tap, 0.0);
    }
}

} // namespace Jetstream::FFT
