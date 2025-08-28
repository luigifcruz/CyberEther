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
    Multiply();
    ~Multiply();

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

    constexpr Taint taint() const {
        if constexpr (D == Device::CPU) {
            return Taint::DISCONTIGUOUS;
        } else {
            // TODO: Implement discontiguous support for Metal.
            return Taint::CLEAN;
        }
    }

    void info() const final;

    // Constructor

    Result create();

 protected:
    Result createCompute(const Context& ctx) final;
    Result compute(const Context& ctx) final;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_MULTIPLY_CPU_AVAILABLE
JST_MULTIPLY_CPU(JST_SPECIALIZATION);
#endif
#ifdef JETSTREAM_MODULE_MULTIPLY_METAL_AVAILABLE
JST_MULTIPLY_METAL(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
