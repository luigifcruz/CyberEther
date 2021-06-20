#ifndef JETSTREAM_FFT_GENERIC_H
#define JETSTREAM_FFT_GENERIC_H

#include "jetstream/base.hpp"

namespace Jetstream {
namespace FFT {

using TI = nonstd::span<std::complex<float>>;
using TO = nonstd::span<float>;

struct Config {
    Range<float> amplitude = {-200.0f, 0.0f};

    Data<TI> input0;
    Jetstream::Policy policy;
};

class Generic : public Module {
public:
    explicit Generic(const Config &);
    virtual ~Generic() = default;

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

} // namespace FFT
} // namespace Jetstream

#endif
