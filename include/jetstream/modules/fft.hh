#ifndef JETSTREAM_MODULES_FFT_HH
#define JETSTREAM_MODULES_FFT_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

#ifdef JETSTREAM_MODULE_FFT_CPU_AVAILABLE
#include "jetstream/tools/pocketfft.hh"
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

#define JST_FFT_CPU(MACRO) \
    MACRO(FFT, CPU, CF32)

#define JST_FFT_METAL(MACRO) \
    MACRO(FFT, Metal, CF32)

template<Device D, typename T = CF32>
class FFT : public Module, public Compute {
 public:
    // Configuration 

    struct Config {
        bool forward = true;

        JST_SERDES(forward);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, T> buffer;

        JST_SERDES_INPUT(buffer);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, T> buffer;

        JST_SERDES_OUTPUT(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, T>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    void info() const final;

    // Constructor

    Result create();

 protected:
    Result createCompute(const RuntimeMetadata& meta) final;
    Result destroyCompute(const RuntimeMetadata& meta) final;
    Result compute(const RuntimeMetadata& meta) final;

 private:
#ifdef JETSTREAM_MODULE_FFT_CPU_AVAILABLE
    struct {
        pocketfft::shape_t shape;
        pocketfft::stride_t stride;
        pocketfft::shape_t axes;
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

    U64 numberOfOperations = 0;
    U64 numberOfElements = 0;
    U64 elementStride = 0;

    JST_DEFINE_IO();
};

#ifdef JETSTREAM_MODULE_FFT_CPU_AVAILABLE
JST_FFT_CPU(JST_SPECIALIZATION);
#endif
#ifdef JETSTREAM_MODULE_FFT_METAL_AVAILABLE
JST_FFT_METAL(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
