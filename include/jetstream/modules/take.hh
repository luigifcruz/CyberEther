#ifndef JETSTREAM_MODULES_TAKE_HH
#define JETSTREAM_MODULES_TAKE_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_TAKE_CPU(MACRO) \
    MACRO(Take, CPU, CF32) \
    MACRO(Take, CPU, F32)

template<Device D, typename T = CF32>
class Take : public Module, public Compute {
 public:
    // Configuration 

    struct Config {
        U64 index = 0;
        U64 axis = 0;

        JST_SERDES(index, axis);
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
    Result compute(const Context& ctx) final;

 private:
    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_TAKE_CPU_AVAILABLE
JST_TAKE_CPU(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
