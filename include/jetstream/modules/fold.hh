#ifndef JETSTREAM_MODULES_FOLD_HH
#define JETSTREAM_MODULES_FOLD_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory2/tensor.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_FOLD_CPU(MACRO) \
    MACRO(Fold, CPU, CF32) \
    MACRO(Fold, CPU, F32)

template<Device D, typename T = CF32>
class Fold : public Module, public Compute {
 public:
    Fold();
    ~Fold();

    // Configuration

    struct Config {
        U64 axis = 0;
        U64 offset = 0;
        U64 size = 0;

        JST_SERDES(axis, offset, size);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        mem2::Tensor buffer;

        JST_SERDES_INPUT(buffer);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        mem2::Tensor buffer;

        JST_SERDES_OUTPUT(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const mem2::Tensor& getOutputBuffer() const {
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
    struct Impl;
    std::unique_ptr<Impl> impl;

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_FOLD_CPU_AVAILABLE
JST_FOLD_CPU(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
