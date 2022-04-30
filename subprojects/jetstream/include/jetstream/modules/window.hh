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
        U64 size;
    };

    struct Input {
    };

    struct Output {
        Memory::Vector<D, T> window;
    };

    explicit Window(const Config&, const Input&);

    constexpr const U64 getWindowSize() const {
        return this->config.size;
    }

    constexpr const Memory::Vector<D, T>& getWindowBuffer() const {
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
