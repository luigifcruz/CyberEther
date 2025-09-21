#ifndef JETSTREAM_MODULES_ADD_HH
#define JETSTREAM_MODULES_ADD_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_ADD_CPU(MACRO) \
    MACRO(Add, CPU, CF32) \
    MACRO(Add, CPU, F32)

template<Device D, typename T = CF32>
class Add : public Module, public Compute {
 public:
    Add();
    ~Add();

    // Configuration

    struct Config {
        JST_SERDES();
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, T> addendA;
        Tensor<D, T> addendB;

        JST_SERDES_INPUT(addendA, addendB);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, T> sum;

        JST_SERDES_OUTPUT(sum);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, T>& getOutputSum() const {
        return this->output.sum;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    constexpr Taint taint() const {
        if constexpr (D == Device::CPU) {
            return Taint::DISCONTIGUOUS;
        } else {
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

#ifdef JETSTREAM_MODULE_ADD_CPU_AVAILABLE
JST_ADD_CPU(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif