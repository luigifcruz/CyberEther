#ifndef JETSTREAM_MODULES_FFT_BASE_HH
#define JETSTREAM_MODULES_FFT_BASE_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"

#ifdef JETSTREAM_FFT_CPU_AVAILABLE
#include <fftw3.h>
#endif

namespace Jetstream {

template<Device D>
class FFT : public Module {
 public:
    struct Config {
        Direction direction = Direction::Forward;
        Range<F64> amplitude = {-200.0f, 0.0f};
    };

    struct Input {
        const Vector<D, CF32>& buffer;
    };

    struct Output {
        Vector<D, CF32> buffer;
    };

    explicit FFT(const Config&, const Input&);

    constexpr Range<float> amplitude() const {
        return config.amplitude;
    }

    Range<float> amplitude(const Range<float>& amplitude) {
        config.amplitude = amplitude;
        return this->amplitude();
    }

    constexpr const Vector<D, CF32>& getOutputBuffer() const {
        return this->output.buffer;
    }

    constexpr const Config getConfig() const {
        return config;
    }

 protected:
    const Result compute() final;

 private:
    const Config config;
    const Input input;
    Output output;

#ifdef JETSTREAM_FFT_CPU_AVAILABLE
    struct {
        fftwf_plan fftPlan;
        std::vector<CF32> fftWindow;
    } CPU;
#endif
};

}  // namespace Jetstream

#endif
