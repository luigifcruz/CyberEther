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
    struct Config {
        std::array<U64, 2> shape;
    };

    struct Input {
    };

    struct Output {
        Vector<D, T, 2> window;
    };

    explicit Window(const Config& config,
                    const Input& input);

    constexpr const Device device() const {
        return D;
    }

    const std::string name() const {
        return "Window";
    }

    void summary() const final;

    constexpr const Vector<D, T, 2>& getWindowBuffer() const {
        return this->output.window;
    }

    constexpr const Config getConfig() const {
        return config;
    }

 private:
    const Config config;
    const Input input;
    Output output;
};

}  // namespace Jetstream

#endif
