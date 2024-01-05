#ifndef JETSTREAM_MODULES_SCALE_HH
#define JETSTREAM_MODULES_SCALE_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_SCALE_CPU(MACRO) \
    MACRO(Scale, CPU, F32)

#define JST_SCALE_METAL(MACRO) \
    MACRO(Scale, Metal, F32)

template<Device D, typename T = F32>
class Scale : public Module, public Compute {
 public:
    // Configuration 

    struct Config {
        Range<T> range = {-1.0, +1.0};

        JST_SERDES(range);
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
    
    // Miscellaneous

    constexpr const Range<T>& range() const {
        return this->config.range;
    }

    const Range<T>& range(const Range<T>& range) {
        this->config.range = range;
        return range;
    }

 protected:
    Result createCompute(const RuntimeMetadata& meta) final;
    Result compute(const RuntimeMetadata& meta) final;

 private:
#ifdef JETSTREAM_MODULE_MULTIPLY_METAL_AVAILABLE
    struct MetalConstants {
        F32 min;
        F32 max;
    };

    struct {
        MTL::ComputePipelineState* state;
        Tensor<Device::Metal, U8> constants;
    } metal;
#endif

    JST_DEFINE_IO();
};

#ifdef JETSTREAM_MODULE_SCALE_CPU_AVAILABLE
JST_SCALE_CPU(JST_SPECIALIZATION);
#endif
#ifdef JETSTREAM_MODULE_SCALE_METAL_AVAILABLE
JST_SCALE_METAL(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
