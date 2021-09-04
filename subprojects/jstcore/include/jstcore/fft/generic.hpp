#ifndef JETSTREAM_FFT_GENERIC_H
#define JETSTREAM_FFT_GENERIC_H

#include "jetstream/base.hpp"

namespace Jetstream::FFT {

class Generic : public Module {
public:
    struct Config {
        Range<float> amplitude = {-200.0f, 0.0f};
    };

    struct Input {
        Data<VCF32> in;
    };

    constexpr Data<nonstd::span<float>>& output() {
        return out;
    };

    explicit Generic(const Config &, const Input &);
    virtual ~Generic() = default;

    Result compute();

    constexpr Range<float> amplitude() const {
        return config.amplitude;
    }
    Range<float> amplitude(const Range<float> &);

protected:
    Config config;
    const Input input;
    Data<VF32> out;

    std::vector<std::complex<float>> window;
};

} // namespace Jetstream::FFT

#endif
