#ifndef JETSTREAM_MODULES_FFT_HH
#define JETSTREAM_MODULES_FFT_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"
#include "jetstream/graph/base.hh"

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
        Direction direction = Direction::Forward;
    };

    struct Input {
        const Vector<D, T, 2>& buffer;
    };

    struct Output {
        Vector<D, T, 2> buffer;
    };

    explicit FFT(const Config& config, 
                 const Input& input); 

    constexpr Device device() const {
        return D;
    }

    const std::string name() const {
        return "Fast-Fourier Transform";
    }

    void summary() const final;

    constexpr const Vector<D, T, 2>& getOutputBuffer() const {
        return this->output.buffer;
    }

    constexpr Config getConfig() const {
        return config;
    }

 protected:
    Result createCompute(const RuntimeMetadata& meta) final;
    Result destroyCompute(const RuntimeMetadata& meta) final;
    Result compute(const RuntimeMetadata& meta) final;

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
        VkFFTApplication* app;
        VkFFTConfiguration* configuration;
        const MTL::Buffer* input;
        MTL::Buffer* output;
    } metal;
#endif
};

}  // namespace Jetstream

#endif
