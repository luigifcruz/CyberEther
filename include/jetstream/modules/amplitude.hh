#ifndef JETSTREAM_MODULES_AMPLITUDE_HH
#define JETSTREAM_MODULES_AMPLITUDE_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_AMPLITUDE_CPU(MACRO) \
    MACRO(Amplitude, CPU, CF32, F32) \
    MACRO(Amplitude, CPU, F32, F32)

#define JST_AMPLITUDE_METAL(MACRO) \
    MACRO(Amplitude, Metal, CF32, F32) \
    MACRO(Amplitude, Metal, F32, F32)

#define JST_AMPLITUDE_CUDA(MACRO) \
    MACRO(Amplitude, CUDA, CF32, F32) \
    MACRO(Amplitude, CUDA, F32, F32)

template<Device D, typename IT = CF32, typename OT = F32>
class Amplitude : public Module, public Compute {
 public:
    Amplitude();
    ~Amplitude();

    // Configuration 

    struct Config {
        JST_SERDES();
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
    Result compute(const Context& ctx) final;

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;

    F32 scalingCoeff = 0.0f;
    U64 numberOfElements = 0;

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_AMPLITUDE_CPU_AVAILABLE
JST_AMPLITUDE_CPU(JST_SPECIALIZATION);
#endif
#ifdef JETSTREAM_MODULE_AMPLITUDE_CUDA_AVAILABLE
JST_AMPLITUDE_CUDA(JST_SPECIALIZATION);
#endif
#ifdef JETSTREAM_MODULE_AMPLITUDE_METAL_AVAILABLE
JST_AMPLITUDE_METAL(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
