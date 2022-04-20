#ifndef JSTCORE_FFT_GENERIC_H
#define JSTCORE_FFT_GENERIC_H

#include "jetstream/base.hpp"

namespace Jetstream::FFT {

template<Device D> class Backend;

class Generic : public Module {
public:
    struct Config {
        Range<float> amplitude = {-200.0f, 0.0f};
    };

    struct Input {
        const Data<VCF32> in;
    };

    constexpr const Data<VF32>& output() {
        return out;
    };

    explicit Generic(const Config&, const Input&);
    virtual ~Generic() = default;

    constexpr Range<float> amplitude() const {
        return config.amplitude;
    }
    Range<float> amplitude(const Range<float>&);

protected:
    Config config;
    const Input input;
    Data<VF32> out;

    const Result compute();

    virtual const Result underlyingCompute() = 0;

    std::vector<std::complex<float>> window;
};

} // namespace Jetstream::FFT

#endif
