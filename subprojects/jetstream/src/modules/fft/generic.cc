#include "jetstream/modules/fft/generic.hpp"

namespace Jetstream {

FFT::FFT(const Config & cfg, Connections& input) : Module(input), cfg(cfg) {
    in = getInput<Data<TI>>("input0");

    auto n = in.buf.size();
    window.resize(in.buf.size());

    float tap;
    for (size_t i = 0; i < n; i++) {
        tap = 0.5 * (1 - cos(2 * M_PI * i / n));
        tap = (i % 2) == 0 ? tap : -tap;
        window[i] = std::complex<float>(tap, 0.0);
    }
}

Range<float> FFT::amplitude(const Range<float> & ampl) {
    cfg.amplitude = ampl;

    return this->amplitude();
}

} // namespace Jetstream
