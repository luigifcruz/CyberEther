#ifndef JETSTREAM_MODULES_PAD_HH
#define JETSTREAM_MODULES_PAD_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory2/tensor.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_PAD_CPU(MACRO) \
    MACRO(Pad, CPU, CF32) \
    MACRO(Pad, CPU, F32)

template<Device D, typename T = CF32>
class Pad : public Module, public Compute {
 public:
    Pad();
    ~Pad();

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
        mem2::Tensor unpadded;

        JST_SERDES_INPUT(unpadded);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        mem2::Tensor padded;

        JST_SERDES_OUTPUT(padded);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const mem2::Tensor& getOutputPadded() const {
        return this->output.padded;
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
    std::unique_ptr<Impl> impl;

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_PAD_CPU_AVAILABLE
JST_PAD_CPU(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
