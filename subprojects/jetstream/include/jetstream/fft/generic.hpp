#ifndef JETSTREAM_FFT_GENERIC_H
#define JETSTREAM_FFT_GENERIC_H

#include "jetstream/module.hpp"

namespace Jetstream {

class FFT : public Module {
public:
    using TI = nonstd::span<std::complex<float>>;
    using TO = nonstd::span<float>;

#ifdef JETSTREAM_FFT_FFTW_AVAILABLE
    class CPU;
#endif
#ifdef JETSTREAM_FFT_CUDA_AVAILABLE
    class CUDA;
#endif

    struct Config {
        Range<float> amplitude = {-200.0f, 0.0f};

        Data<TI> input0;
        Jetstream::Policy policy;
    };

    explicit FFT(const Config &);
    virtual ~FFT() = default;

    constexpr Range<float> amplitude() const {
        return cfg.amplitude;
    }
    Range<float> amplitude(const Range<float> &);

    constexpr Data<TO> output() const {
        return out;
    }

protected:
    Config cfg;
    Data<TI> in;
    Data<TO> out;

    std::vector<std::complex<float>> window;
};

} // namespace Jetstream

#endif
