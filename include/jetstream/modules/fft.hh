#ifndef JETSTREAM_MODULES_FFT_HH
#define JETSTREAM_MODULES_FFT_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_FFT_CPU(MACRO) \
    MACRO(FFT, CPU, CF32, CF32) \
    MACRO(FFT, CPU, F32, CF32)

#define JST_FFT_METAL(MACRO) \
    MACRO(FFT, Metal, CF32, CF32)

#define JST_FFT_CUDA(MACRO) \
    MACRO(FFT, CUDA, CF32, CF32)

template<Device D, typename IT = CF32, typename OT = CF32>
class FFT : public Module, public Compute {
 public:
    FFT();
    ~FFT();

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
        Tensor<D, IT> buffer;

        JST_SERDES_INPUT(buffer);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, OT> buffer;

        JST_SERDES_OUTPUT(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, OT>& getOutputBuffer() const {
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
    Result createCompute(const Context& ctx) final;
    Result destroyCompute(const Context& ctx) final;
    Result compute(const Context& ctx) final;

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;

    U64 numberOfOperations = 0;
    U64 numberOfElements = 0;
    U64 elementStride = 0;

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_FFT_CPU_AVAILABLE
JST_FFT_CPU(JST_SPECIALIZATION);
#endif
#ifdef JETSTREAM_MODULE_FFT_CUDA_AVAILABLE
JST_FFT_CUDA(JST_SPECIALIZATION);
#endif
#ifdef JETSTREAM_MODULE_FFT_METAL_AVAILABLE
JST_FFT_METAL(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
