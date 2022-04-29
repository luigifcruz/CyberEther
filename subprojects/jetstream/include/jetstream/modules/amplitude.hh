#ifndef JETSTREAM_MODULES_AMPLITUDE_HH
#define JETSTREAM_MODULES_AMPLITUDE_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"

namespace Jetstream {

template<Device D>
class Amplitude : public Module {
 public:
    struct Config {
        U64 size;
    };

    struct Input {
        const Vector<D, CF32>& buffer;
    };

    struct Output {
        Vector<D, F32> buffer;
    };

    explicit Amplitude(const Config&, const Input&);

    constexpr const U64 getBufferSize() const {
        return this->config.size;
    }

    constexpr const Vector<D, F32>& getOutputBuffer() const {
        return this->output.buffer;
    }

    constexpr const Config getConfig() const {
        return config;
    }

 protected:
    const Result compute() final;

 private:
    const Config config;
    const Input input;
    Output output;
};

}  // namespace Jetstream

#endif
