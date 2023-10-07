#ifndef JETSTREAM_MODULES_WINDOW_HH
#define JETSTREAM_MODULES_WINDOW_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"

#include "jetstream/memory/base.hh"

namespace Jetstream {

template<Device D, typename T = CF32>
class Window : public Module {
 public:
    // Configuration 

    struct Config {
        VectorShape<2> shape;

        JST_SERDES(
            JST_SERDES_VAL("shape", shape);
        );
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        JST_SERDES();
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Vector<D, T, 2> window;

        JST_SERDES(
            JST_SERDES_VAL("window", window);
        );
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Vector<D, T, 2>& getWindowBuffer() const {
        return this->output.window;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string_view name() const {
        return "window";
    }

    std::string_view prettyName() const {
        return "Window";
    }

    void summary() const final;

    // Constructor

    Result create();

    JST_DEFINE_IO();
};

}  // namespace Jetstream

#endif
