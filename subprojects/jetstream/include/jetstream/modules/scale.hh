#ifndef JETSTREAM_MODULES_SCALE_HH
#define JETSTREAM_MODULES_SCALE_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"

namespace Jetstream {

template<Device D, typename T = F32>
class Scale : public Module {
 public:
    struct Config {
        U64 size;
        Range<T> range = {-1.0, +1.0};
    };

    struct Input {
        const Vector<D, T>& buffer;
    };

    struct Output {
        Vector<D, T> buffer;
    };

    explicit Scale(const Config&, const Input&);

    constexpr const U64 getBufferSize() const {
        return this->config.size;
    }

    constexpr const Vector<D, T>& getOutputBuffer() const {
        return this->output.buffer;
    }

    constexpr const Config getConfig() const {
        return this->config;
    }

    constexpr const Range<T>& range() const {
        return this->config.range;
    }

    const Range<T>& range(const Range<T>& range) {
        this->config.range = range;
        return this->range();
    }

 protected:
    const Result compute(const RuntimeMetadata& meta = {}) final;

 private:
    Config config;
    const Input input;
    Output output;
};

}  // namespace Jetstream

#endif
