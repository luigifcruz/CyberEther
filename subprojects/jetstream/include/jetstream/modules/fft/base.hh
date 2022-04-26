#ifndef JETSTREAM_MODULES_FFT_BASE_HH
#define JETSTREAM_MODULES_FFT_BASE_HH

#include "jetstream/base.hh"
#include "jetstream/module.hh"
#include "jetstream/memory/base.hh"

namespace Jetstream {

template<Device D>
class FFT : public Module {
 public:
    enum class Direction : I64 {
        Forward = 1,
        Backward = -1,
    };

    struct Config {
        Direction direction = Direction::Forward;
    };

    struct Input {
        const Vector<D, CF32>& buffer;
    };

    struct Output {
        Vector<D, CF32> buffer;
    };

    explicit FFT(const Config&, const Input&);

    constexpr const Vector<D, CF32>& getOutputBuffer() const {
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