#ifndef JETSTREAM_FFT_GENERIC_H
#define JETSTREAM_FFT_GENERIC_H

#include "jetstream/modules/module.hpp"

namespace Jetstream {

class FFT : public Module {
public:
    using TI = VCF32;
    using TO = VF32;

    struct Config {
        Range<float> amplitude = {-200.0f, 0.0f};
    };

    static Connections inputBlueprint(const Locale & device) {
        switch (device) {
            case Locale::CPU:
            return {
                {"input0", Data<TI>{Locale::CPU, {}}},
            };
            case Locale::CUDA:
            return {
                {"input0", Data<TI>{Locale::CUDA, {}}},
            };
        }
        return {};
    }

#ifdef JETSTREAM_FFT_FFTW_AVAILABLE
    class CPU;
#endif
#ifdef JETSTREAM_FFT_CUDA_AVAILABLE
    class CUDA;
#endif

    explicit FFT(const Config & cfg, Connections& input);
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
