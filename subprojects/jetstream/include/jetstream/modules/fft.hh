#ifndef JETSTREAM_MODULES_FFT_HH
#define JETSTREAM_MODULES_FFT_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"

#ifdef JETSTREAM_MODULE_FFT_CPU_AVAILABLE
#include <fftw3.h>
#endif

#ifdef JETSTREAM_MODULE_FFT_METAL_AVAILABLE
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#define VKFFT_BACKEND 5
#include "jetstream/tools/vkFFT.h"
#pragma GCC diagnostic pop
#endif

namespace Jetstream {

template<Device D, typename T = CF32>
class FFT : public Module, public Compute {
 public:
    struct Config {
        U64 size;
        Direction direction = Direction::Forward;
    };

    struct Input {
        // TODO: Change back.
        const Vector<Device::CPU, T>& buffer;
    };

    struct Output {
        // TODO: Change back.
        Vector<Device::CPU, T> buffer;
    };

    explicit FFT(const Config& config, 
                 const Input& input); 

    constexpr const Device device() const {
        return D;
    }

    constexpr const Taint taints() const {
        return Taint::None;
    }

    void summary() const final;

    constexpr const U64 getBufferSize() const {
        return this->config.size;
    }

    // TODO: Change back.
    constexpr const Vector<Device::CPU, T>& getOutputBuffer() const {
        return this->output.buffer;
    }

    constexpr const Config getConfig() const {
        return config;
    }

 protected:
    const Result createCompute(const RuntimeMetadata& meta) final;
    const Result compute(const RuntimeMetadata& meta) final;

 private:
    const Config config;
    const Input input;
    Output output;

#ifdef JETSTREAM_MODULE_FFT_CPU_AVAILABLE
    struct {
        fftwf_plan fftPlanCF32;
        fftw_plan fftPlanCF64;
    } cpu;
#endif

#ifdef JETSTREAM_MODULE_FFT_METAL_AVAILABLE
    struct {
        MTL::Buffer* input;
        MTL::Buffer* output;
        VkFFTApplication* app;
    } metal;
#endif
};

}  // namespace Jetstream

#endif
