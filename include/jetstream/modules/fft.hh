#ifndef JETSTREAM_MODULES_FFT_HH
#define JETSTREAM_MODULES_FFT_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

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
    // Configuration 

    struct Config {
        bool forward = true;
        U64 offset = 0;
        U64 size = 0;

        JST_SERDES(
            JST_SERDES_VAL("forward", forward);
            JST_SERDES_VAL("offset", offset);
            JST_SERDES_VAL("size", size);
        );
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Vector<D, T, 2> buffer;

        JST_SERDES(
            JST_SERDES_VAL("buffer", buffer);
        );
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Vector<D, T, 2> buffer;

        JST_SERDES(
            JST_SERDES_VAL("buffer", buffer);
        );
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Vector<D, T, 2>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string_view name() const {
        return "fft";
    }

    std::string_view prettyName() const {
        return "Fast-Fourier Transform";
    }

    void summary() const final;

    // Constructor

    Result create();

 protected:
    Result createCompute(const RuntimeMetadata& meta) final;
    Result destroyCompute(const RuntimeMetadata& meta) final;
    Result compute(const RuntimeMetadata& meta) final;

 private:
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

    JST_DEFINE_MODULE_IO();
};

}  // namespace Jetstream

#endif
