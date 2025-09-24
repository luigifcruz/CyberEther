#ifndef JETSTREAM_MODULES_THROTTLE_HH
#define JETSTREAM_MODULES_THROTTLE_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory2/tensor.hh"
#include "jetstream/compute/graph/base.hh"

#include <chrono>

namespace Jetstream {

#define JST_THROTTLE_CPU(MACRO) \
    MACRO(Throttle, CPU, CF32)

template<Device D, typename T = F32>
class Throttle : public Module, public Compute {
 public:
    Throttle();
    ~Throttle();

    // Configuration

    struct Config {
        U64 intervalMs = 100;  // Throttle interval in milliseconds

        JST_SERDES(intervalMs);
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

    // Miscellaneous

    constexpr U64 intervalMs() const {
        return this->config.intervalMs;
    }

    const U64& intervalMs(const U64& intervalMs);

 protected:
    Result compute(const Context& ctx) final;
    Result computeReady() final;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_THROTTLE_CPU_AVAILABLE
JST_THROTTLE_CPU(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
