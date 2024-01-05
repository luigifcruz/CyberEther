#ifndef JETSTREAM_MODULES_AMPLITUDE_HH
#define JETSTREAM_MODULES_AMPLITUDE_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_AMPLITUDE_CPU(MACRO) \
    MACRO(Amplitude, CPU, CF32, F32)

#define JST_AMPLITUDE_METAL(MACRO) \
    MACRO(Amplitude, Metal, CF32, F32)

template<Device D, typename IT = CF32, typename OT = F32>
class Amplitude : public Module, public Compute {
 public:
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
    Result createCompute(const RuntimeMetadata& meta) final;
    Result compute(const RuntimeMetadata& meta) final;

 private:
#ifdef JETSTREAM_MODULE_MULTIPLY_METAL_AVAILABLE
    struct MetalConstants {
        F32 scalingSize;
    };

    struct {
        MTL::ComputePipelineState* state;
        Tensor<Device::Metal, U8> constants;
    } metal;
#endif

    U64 scalingSize = 0;

    JST_DEFINE_IO();
};

#ifdef JETSTREAM_MODULE_AMPLITUDE_CPU_AVAILABLE
JST_AMPLITUDE_CPU(JST_SPECIALIZATION);
#endif
#ifdef JETSTREAM_MODULE_AMPLITUDE_METAL_AVAILABLE
JST_AMPLITUDE_METAL(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
