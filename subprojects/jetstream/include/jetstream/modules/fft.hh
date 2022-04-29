#ifndef JETSTREAM_MODULES_FFT_HH
#define JETSTREAM_MODULES_FFT_HH

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
        U64 size;
        Direction direction = Direction::Forward;
    };

    struct Input {
        const Vector<D, CF32>& buffer;
    };

    struct Output {
        Vector<D, CF32> buffer;
    };

    explicit FFT(const Config&, const Input&);

    constexpr const U64 getBufferSize() const {
        return this->config.size;
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
