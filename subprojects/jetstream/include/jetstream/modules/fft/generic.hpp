#ifndef JETSTREAM_FFT_GENERIC_H
#define JETSTREAM_FFT_GENERIC_H

#include "jetstream/modules/module.hpp"

namespace Jetstream {

class FFT : public Module {
public:
    using TI = VCF32;
    using TO = VF32;

#ifdef JETSTREAM_FFT_FFTW_AVAILABLE
    class CPU;
#endif
#ifdef JETSTREAM_FFT_CUDA_AVAILABLE
    class CUDA;
#endif

    struct Config {
        Range<float> amplitude = {-200.0f, 0.0f};
    };

    explicit FFT(const Config & cfg, IO & input);
    virtual ~FFT() = default;

    constexpr Range<float> amplitude() const {
        return cfg.amplitude;
    }
    Range<float> amplitude(const Range<float> &);

protected:
    Config cfg;
    Data<TI> in;
    Data<TO> out;

    std::vector<std::complex<float>> window;
};

} // namespace Jetstream

#endif
