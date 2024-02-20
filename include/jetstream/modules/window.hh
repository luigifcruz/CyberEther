#ifndef JETSTREAM_MODULES_WINDOW_HH
#define JETSTREAM_MODULES_WINDOW_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"

namespace Jetstream {

#define JST_WINDOW_CPU(MACRO) \
    MACRO(Window, CPU, CF32)

template<Device D, typename T = CF32>
class Window : public Module, public Compute {
 public:
    // Configuration 

    struct Config {
        U64 size;

        JST_SERDES(size);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        JST_SERDES_INPUT();
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, T> window;

        JST_SERDES_OUTPUT(window);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, T>& getOutputWindow() const {
        return this->output.window;
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
    bool baked = false;

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_WINDOW_CPU_AVAILABLE
JST_WINDOW_CPU(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
