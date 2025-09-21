#ifndef JETSTREAM_MODULES_FM_HH
#define JETSTREAM_MODULES_FM_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_FM_CPU(MACRO) \
    MACRO(FM, CPU, CF32, F32)

template<Device D, typename IT = CF32, typename OT = F32>
class FM : public Module, public Compute {
 public:
    FM();
    ~FM();

    // Configuration

    struct Config {
        F32 sampleRate = 240e3f;

        JST_SERDES(sampleRate);
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

#ifdef JETSTREAM_MODULE_FM_CPU_AVAILABLE
JST_FM_CPU(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
