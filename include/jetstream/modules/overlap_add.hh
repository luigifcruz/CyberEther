#ifndef JETSTREAM_MODULES_OVERLAP_ADD_HH
#define JETSTREAM_MODULES_OVERLAP_ADD_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_OVERLAP_ADD_CPU(MACRO) \
    MACRO(OverlapAdd, CPU, CF32) \
    MACRO(OverlapAdd, CPU, F32)

template<Device D, typename T = CF32>
class OverlapAdd : public Module, public Compute {
 public:
    // Configuration 

    struct Config {
        U64 axis = 1;

        JST_SERDES(axis);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, T> buffer;
        Tensor<D, T> overlap;

        JST_SERDES_INPUT(buffer, overlap);
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
    Result createCompute(const Context& ctx) final;
    Result compute(const Context& ctx) final;

    Tensor<D, T> previousOverlap;

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_OVERLAP_ADD_CPU_AVAILABLE
JST_OVERLAP_ADD_CPU(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
