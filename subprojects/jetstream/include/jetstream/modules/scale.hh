#ifndef JETSTREAM_MODULES_SCALE_HH
#define JETSTREAM_MODULES_SCALE_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"

namespace Jetstream {

template<Device D>
class Scale : public Module {
 public:
    struct Config {
        U64 size;
        Range<F64> range = {-1.0f, +1.0f};
    };

    struct Input {
        const Vector<D, F32>& buffer;
    };

    struct Output {
        Vector<D, F32> buffer;
    };

    explicit Scale(const Config&, const Input&);

    constexpr const U64 getBufferSize() const {
        return this->config.size;
    }

    constexpr const Vector<D, F32>& getOutputBuffer() const {
        return this->output.buffer;
    }

    constexpr const Config getConfig() const {
        return config;
    }

    constexpr Range<float> range() const {
        return this->config.range;
    }

    Range<float> range(const Range<float>& range) {
        this->config.range = range;
        return this->range();
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
