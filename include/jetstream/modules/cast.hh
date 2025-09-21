#ifndef JETSTREAM_MODULES_CAST_HH
#define JETSTREAM_MODULES_CAST_HH

#include <algorithm>

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_CAST_CPU(MACRO) \
    MACRO(Cast, CPU, CF32, F32) \
    MACRO(Cast, CPU, CI8, CF32)

template<Device D, typename IT = F32, typename OT = I16>
class Cast : public Module, public Compute {
 public:
    Cast();
    ~Cast();

    // Configuration

    struct Config {
        F32 scaler = 0.0f;

        JST_SERDES(scaler);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, IT> buffer;

        JST_SERDES_INPUT(buffer);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, OT> buffer;

        JST_SERDES_OUTPUT(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, OT>& getOutputBuffer() const {
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

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_CAST_CPU_AVAILABLE
JST_CAST_CPU(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
