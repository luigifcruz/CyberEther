#ifndef JETSTREAM_MODULES_MULTIPLY_HH
#define JETSTREAM_MODULES_MULTIPLY_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_MULTIPLY_CPU(MACRO) \
    MACRO(Multiply, CPU, CF32) \
    MACRO(Multiply, CPU, F32)

#define JST_MULTIPLY_METAL(MACRO) \
    MACRO(Multiply, Metal, CF32)

template<Device D, typename T = CF32>
class Multiply : public Module, public Compute {
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
        Tensor<D, T> factorA;
        Tensor<D, T> factorB;

        JST_SERDES_INPUT(factorA, factorB);
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
    Tensor<D, T> a;
    Tensor<D, T> b;
    Tensor<D, T> c;

#ifdef JETSTREAM_MODULE_MULTIPLY_METAL_AVAILABLE
    struct {
        MTL::ComputePipelineState* state;
    } metal;
#endif

    JST_DEFINE_IO();
};

#ifdef JETSTREAM_MODULE_MULTIPLY_CPU_AVAILABLE
JST_MULTIPLY_CPU(JST_SPECIALIZATION);
#endif
#ifdef JETSTREAM_MODULE_MULTIPLY_METAL_AVAILABLE
JST_MULTIPLY_METAL(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
