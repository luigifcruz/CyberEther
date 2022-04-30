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

template<Device D, typename T = CF32>
class FFT : public Module {
 public:
    struct Config {
        U64 size;
        Direction direction = Direction::Forward;
    };

    struct Input {
        const Memory::Vector<D, T>& buffer;
    };

    struct Output {
        Memory::Vector<D, T> buffer;
    };

    explicit FFT(const Config&, const Input&);

    constexpr const U64 getBufferSize() const {
        return this->config.size;
    }

    constexpr const Memory::Vector<D, T>& getOutputBuffer() const {
        return this->output.buffer;
    }

    constexpr const Config getConfig() const {
        return config;
    }

 protected:
    const Result compute(const RuntimeMetadata& meta = {}) final;

 private:
    const Config config;
    const Input input;
    Output output;

#ifdef JETSTREAM_FFT_CPU_AVAILABLE
    const Result generatePlanCPU();

    struct {
        fftwf_plan fftPlanCF32;
        fftw_plan fftPlanCF64;
    } CPU;
#endif
};

}  // namespace Jetstream

#endif
