#ifndef JETSTREAM_MODULES_UNPAD_HH
#define JETSTREAM_MODULES_UNPAD_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_UNPAD_CPU(MACRO) \
    MACRO(Unpad, CPU, CF32) \
    MACRO(Unpad, CPU, F32)

template<Device D, typename T = CF32>
class Unpad : public Module, public Compute {
 public:
    // Configuration 

    struct Config {
        U64 size = 33;
        U64 axis = 1;

        JST_SERDES(size, axis);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, T> padded;

        JST_SERDES_INPUT(padded);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, T> unpadded;
        Tensor<D, T> pad;

        JST_SERDES_OUTPUT(unpadded, pad);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, T>& getOutputUnpadded() const {
        return this->output.unpadded;
    }

    constexpr const Tensor<D, T>& getOutputPad() const {
        return this->output.pad;
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

    JST_DEFINE_IO();
};

#ifdef JETSTREAM_MODULE_UNPAD_CPU_AVAILABLE
JST_UNPAD_CPU(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
