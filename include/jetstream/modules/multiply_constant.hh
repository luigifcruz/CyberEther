#ifndef JETSTREAM_MODULES_MULTIPLY_CONSTANT_HH
#define JETSTREAM_MODULES_MULTIPLY_CONSTANT_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_MULTIPLY_CONSTANT_CPU(MACRO) \
    MACRO(MultiplyConstant, CPU, CF32) \
    MACRO(MultiplyConstant, CPU, F32) 

#define JST_MULTIPLY_CONSTANT_METAL(MACRO) \
    MACRO(MultiplyConstant, Metal, CF32) \
    MACRO(MultiplyConstant, Metal, F32)

template<Device D, typename T = CF32>
class MultiplyConstant : public Module, public Compute {
 public:
    // Configuration 

    struct Config {
        T constant;

        JST_SERDES(constant);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, T> factor;

        JST_SERDES_INPUT(factor);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, T> product;

        JST_SERDES_OUTPUT(product);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, T>& getOutputProduct() const {
        return this->output.product;
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
        F32 constantReal;
        F32 constantImage;
    };

    struct {
        MTL::ComputePipelineState* state;
        Tensor<Device::Metal, U8> constants;
    } metal;
#endif

    JST_DEFINE_IO();
};

#ifdef JETSTREAM_MODULE_MULTIPLY_CONSTANT_CPU_AVAILABLE
JST_MULTIPLY_CONSTANT_CPU(JST_SPECIALIZATION);
#endif
#ifdef JETSTREAM_MODULE_MULTIPLY_CONSTANT_METAL_AVAILABLE
JST_MULTIPLY_CONSTANT_METAL(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
